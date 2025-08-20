import os
import cv2
import bs4
import json
import time
import signal
import string
import requests
import re
import pyautogui
import pytesseract
import numpy as np
from PIL import Image
from datetime import datetime
import concurrent.futures
from openai import OpenAI
from openai import RateLimitError
import base64
from io import BytesIO
import google.generativeai as genai
from groq import Groq




# Cấu hình API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gpt_token = os.getenv("OPENAI_API_KEY")
groq_token = os.getenv("GROQ_API_KEY")
gemini_token = os.getenv("GEMINI_API_KEY")
model = "openai/gpt-4.1"

# ================== CONFIG ==================
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client = OpenAI(api_key=api_keys)

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=gpt_token,
)

client2 = Groq(
    api_key=groq_token
)




DISC_WEBHK_URL = os.getenv("DISC_WEBHK_URL")
GOOGLE_SEARCH_URL = "https://www.google.com/search?q="

TIMER_POSITION = (343,267)
TIMER_COLOR = (33, 63, 131, 255)
QUESTION_REGION = (290, 340, 535, 714)

NUM_ANSWERS = 3
QUESTION_TIME = 20
CHECK_SECONDS = 5
# ============================================


# ============== UTILS ==============
def sigint_handler(signum, frame):
    global running
    running = False


def detect_color(color, x, y):
    return pyautogui.pixelMatchesColor(x, y, color, tolerance=5)


def get_game_img(img_region):
    return pyautogui.screenshot(region=img_region)


def process_img(im: Image.Image):
    np_img = np.array(im)
    contrast = 0.8
    brightness = -100
    return cv2.addWeighted(np_img, contrast, np_img, 0, brightness)


def get_text(im: Image.Image):
    img = process_img(im)
    return pytesseract.image_to_string(img, lang="vie")


def rem_punc(input_string: str):
    return input_string.translate(str.maketrans("", "", string.punctuation))


# ============== MODEL ==============
class Question:
    def __init__(self, q="", a=None, num=0):
        self._number = num
        self._question = q
        self._answers = a if a else []

    def __repr__(self):
        return f"Question({self._number}, {self._question}, {self._answers})"

    def get_gpt_prompt(self):
        return f"""{self._question} (pick from {len(self._answers)} options):
{self.format_answers()}
Answer:"""

    def format_answers(self):
        return "\n".join(f"{i}. {ans}" for i, ans in enumerate(self._answers, 1))

    def print(self):
        print(f"Question {self._number}: {self._question}")
        print(self.format_answers())

    def get_json(self):
        return {"number": self._number, "question": self._question, "answers": self._answers}


# ============== GOOGLE HELPER ==============
def gen_google_search(q: Question):
    f_answers = '" OR "'.join(q._answers)
    url = f'{GOOGLE_SEARCH_URL}{q._question} "{f_answers}"'
    return url.replace(" ", "+")


def make_google_soup(url):
    r = requests.get(url)
    r.raise_for_status()
    return bs4.BeautifulSoup(r.text, "lxml")


def count_answers(soup, answers):
    results = dict.fromkeys(answers, 0)
    for item in soup.find_all("div"):
        text = item.get_text().lower()
        text_no_punc = rem_punc(text)
        for ans in answers:
            ans_l = ans.lower()
            if ans_l in text or ans_l in text_no_punc:
                results[ans] += 1
    return results


def get_google_results(q: Question):
    soup = make_google_soup(gen_google_search(q))
    return count_answers(soup, q._answers)

# ============== DUCKDUCKGO HELPER ==============

def duckduckgo_search(q: Question):
    f_answers = '" OR "'.join(q._answers)
    query = f'{q._question} "{f_answers}"'
    query = query.replace(" ", "+")
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def count_answers_duckduckgo(q: Question):
    data = duckduckgo_search(q)
    text = (data.get("AbstractText", "") + " " + data.get("Heading", "")).lower()

    results = dict.fromkeys(q._answers, 0)
    for ans in q._answers:
        ans_l = ans.lower()
        results[ans] += len(re.findall(rf"\b{re.escape(ans_l)}\b", text))
    return results


def get_duckduckgo_results(q: Question):
    return count_answers_duckduckgo(q)
# ============== OPENAI HELPER ==============

def get_gpt_ans(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            top_p=0.2
        )
        return response.choices[0].message.content.strip()

    except RateLimitError as e:
        print("[OpenAI] Rate limit reached, fallback to Google")
        return ""

    except Exception as e:
        print(f"[OpenAI] Error: {e}")
        return ""

# ============== GROQ HELPER ==============


def get_groq_ans(prompt: str):
    try:
        response = client2.chat.completions.create(
            model="openai/gpt-oss-20b",  # hoặc mixtral-8x7b-32768
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=512,
            top_p=1
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Groq] Error: {e}")
        return ""

def get_gpt_ans_with_image(question: Question, image: Image.Image):
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = question.get_gpt_prompt()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # model hỗ trợ vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenAI Vision] Error: {e}")
        return ""
    
    
def get_gemini_ans(prompt: str):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")  # có thể đổi sang "gemini-1.5-pro"
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return ""
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return ""


# ============== MAIN LOGIC ==============
def get_question(raw_text: str, q_num: int, num_ans: int):

    prompt_format_question = f"""
You are an OCR text cleaner and formatter. Your task is to take noisy OCR text as input and produce a clean, well-formatted question and multiple-choice answers.\n\nInput:\n{raw_text}\n\nCleaning rules:\n1. Remove all irrelevant symbols, numbers, or broken characters from OCR (such as ›, {',' }, _, -, etc.).\n2. Fix broken words and merge lines correctly so the text becomes natural.\n3. Add proper punctuation and spacing in Vietnamese.\n4. Keep exactly one question followed by three answers.\n5. Format output clearly as:\n\nCâu hỏi: [nội dung câu hỏi]\n\nA. [đáp án 1]\nB. [đáp án 2]\nC. [đáp án 3]\n\nMake sure the output remains in Vietnamese.

"""
    formatted_text = ""
    print(f"Raw text: {raw_text}")
    if DISC_WEBHK_URL:
                post_disc_web_hk_raw(DISC_WEBHK_URL, raw_text)
                print("🚀 Pushed raw text to discord!")
    
    try: 
        formatted_text = get_groq_ans(prompt_format_question)
    except Exception as e: 
        print(f"[GROQ] Error: {e}")
    if formatted_text == "" :
        formatted_text = raw_text
    split_s = list(filter(None, formatted_text.split("\n")))
    question = " ".join(split_s[:-num_ans]).replace('"', "")
    answers = split_s[-num_ans:]
    return Question(question, answers, q_num)

def get_all_answers(q: Question, img=None):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "google": executor.submit(get_google_results, q),
            "gpt": executor.submit(get_gpt_ans, q.get_gpt_prompt()),
            "groq": executor.submit(get_groq_ans, q.get_gpt_prompt()),
            # "gemini": executor.submit(get_gemini_ans, q.get_gpt_prompt()),
        }

        concurrent.futures.wait(futures.values())

    results = (
        f"Google results: {futures['google'].result()}\n\n"
        f"GPT answer: {futures['gpt'].result()}\n\n"
        f"Groq results: {futures['groq'].result()}\n\n"
        # f"Gemini results: {futures['gemini'].result()}"
    )

    return results




# ========== DISCORD WEBHOOK ============
def create_disc_embed(q: Question, results: str):
    ans_choices = "\n".join(
        f"{i}. [{ans}]({(GOOGLE_SEARCH_URL + ans).replace(' ', '+')})"
        for i, ans in enumerate(q._answers, 1)
    )
    return {
        "title": f"Question {q._number}: {q._question}",
        "url": gen_google_search(q),
        "description": f"{ans_choices}\n\n**{results}**\n",
        "color": 0x000000,
    }


def post_disc_web_hk(url, q: Question, r: str):
    data = {"username": "Momo Trivia", "embeds": [create_disc_embed(q, r)]}
    requests.post(url, json=data)
    


def create_disc_embed_raw(raw_question: str):
    return {
        "title": f"Question: {raw_question}",
        "url": GOOGLE_SEARCH_URL + raw_question.replace(" ", "+"),
        "description": "",
        "color": 0x000000,
    }

def post_disc_web_hk_raw(url, raw_question: str):
    data = {"username": "Momo Trivia", "embeds": [create_disc_embed_raw(raw_question)]}
    requests.post(url, json=data)

# ========== LOG ============
def log_questions(q_list):
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"logs/log_{cur_time}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_name, "w+", encoding="utf-8") as f:
        json.dump([q.get_json() for q in q_list], f, indent=4, ensure_ascii=False)
    print(f"\nSaved questions to {log_name}")


def run():
    q_list = []
    global running
    running = True
    signal.signal(signal.SIGINT, sigint_handler)

    print("waiting for question...\n")
    print("Nhấn 's' để lấy câu hỏi, 'q' để thoát.\n")
    while running:
        user_input = input(">> ").strip().lower()
        if user_input == "q":
            running = False
            break
        elif user_input == "s":
            # Chụp ảnh câu hỏi
            img = get_game_img(QUESTION_REGION)
            screen_text = get_text(img)
            question = get_question(screen_text, len(q_list) + 1, NUM_ANSWERS)

            question.print()
            if DISC_WEBHK_URL:
                post_disc_web_hk(DISC_WEBHK_URL, question, None)
                print("🚀 Pushed question to discord!")
            results = get_all_answers(question, img) if client.api_key else f"Google results: {get_google_results(question)}"
            print(results)

            q_list.append(question)
            if DISC_WEBHK_URL:
                post_disc_web_hk(DISC_WEBHK_URL, question, results)
                print("🚀 Pushed answer to discord!")

            print("\nNhấn 's' để lấy câu hỏi tiếp theo, 'q' để thoát.\n")

    log_questions(q_list)


def main():
    run()


if __name__ == "__main__":
    main()
