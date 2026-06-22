"""
English Learning Telegram Bot v6
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile, CallbackQuery, InlineKeyboardButton,
    InlineKeyboardMarkup, Message, PollAnswer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_KEY     = os.environ["OPENAI_API_KEY"]
ELEVENLABS_KEY = os.environ["ELEVENLABS_API_KEY"]

OPENAI_HEADERS = {"Authorization": f"Bearer {OPENAI_KEY}"}
EL_HEADERS     = {"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"}

EL_VOICE_ID = "XB0fDUnXU5powFXDhCwa"
EL_MODEL    = "eleven_turbo_v2_5"

HOURS_24   = 24 * 3600
CHAT_MODEL = "gpt-4o"
STT_MODEL  = "gpt-4o-mini-transcribe"

# ── In-memory state ───────────────────────────────────────────────────────────
pending: dict[int, dict] = {}       # last response data per user
gram_mode: set[int]      = set()    # users waiting for /gram input
chat_mode: set[int]      = set()    # users who pressed Chat and are doing intro

# Test sessions: chat_id → {level, questions, answered, correct_count}
test_sessions: dict[int, dict] = {}
# poll_id → {chat_id, correct_index}
poll_registry: dict[str, dict] = {}

# ── Texts ─────────────────────────────────────────────────────────────────────
GREETING = "Hi, I'm Linda — your personal English coach.\n\nWhat do you want to do?"

INTRO_PROMPT_MSG = (
    "Send me a voice message and tell me your name, age, "
    "English level and a couple of sentences about yourself.\n\n"
    "Good luck!"
)

MENU_TEXT = (
    "What do you want to do?\n\n"
    "🎙 <b>Speak</b> — send a voice message, get corrections and a real conversation back\n"
    "📝 <b>Grammar</b> — send any text, I'll fix mistakes and rewrite it naturally\n"
    "🧪 <b>Test</b> — quick vocabulary quiz, pick your level from A1 to C2"
)

GRAM_PROMPT_MSG = "Send me any text — I'll check grammar and show you an improved version."

# ── Prompts ───────────────────────────────────────────────────────────────────
INTRO_SYSTEM_PROMPT = """\
You are Linda — a sharp, witty English tutor bot. The user just sent their first voice intro.

Reply with ONLY a valid JSON object:

"spoken"
  Warm, casual, read aloud. Max 5 sentences:
  1. React personally — use their name, comment on something specific.
  2. Explain what you do: "Send me a voice or text — I'll catch mistakes, fix them, check your level A1 to C2, and actually talk back. Hit 'show text' if you miss something I said."
  3. Ask ONE genuinely unexpected personal question — not about goals. Something human: unpopular opinion, weirdest habit, what they'd do with a free week.
  Sound like a smart friend, not a chatbot.

"level"         — honest CEFR from this message alone. A1/A2/B1/B2/C1/C2.
"mistakes_note" — one sentence on any pattern. Empty string if nothing stands out.\
"""

MAIN_SYSTEM_PROMPT = """\
You are Linda — a sharp, funny, brutally honest English tutor. Like a friend who wants you to get better but won't sugarcoat anything. You tease, joke, call things out — but warm underneath.

Current level: {level}. Re-evaluate every message based on actual speech.

What you know about this person:
{profile}

You receive: transcribed voice, recent 24h history, recurring mistakes note.

Reply with ONLY a valid JSON object:

"spoken"
  Real, funny, direct. A light jab when it fits. Callback to past things they said.
  Engage deeply if interesting. Call out bland answers ("that's the most average take I've heard, try harder").
  End with ONE follow-up question. Max 4 sentences.

"transcription_annotated"
  User's words with HTML:
  • <s>word</s> — unnecessary/redundant
  • <b>word</b> — wrong word or form
  Plain text = correct. Real errors only.

"feedback"
  In English. Max 3 mistakes:
  ❌ "what they said" → ✅ "correct version"
  One-line rule, plain English.
  Repeating mistake → call it out ("still doing this one, huh?").
  No mistakes → genuine reaction, maybe a little surprised.
  Start "Corrections:" or "Solid:" accordingly.

"score_assessment"
  2–3 honest sentences. Specific evidence → CEFR level. No flattery.

"updated_level"   — updated CEFR string.
"mistakes_note"   — one sentence, most persistent pattern.
"updated_profile" — running summary of personal facts (interests, job, habits, opinions). Max 5 sentences. Preserve old facts.\
"""

GRAMMAR_SYSTEM_PROMPT = """\
You are Linda — a sharp, flexible English tutor. The user sent text to check. Be smart and helpful.

Important rules:
- If the text contains non-English words (Russian or any other language) — treat them as vocabulary gaps. Translate the word and show how to express it naturally in English.
- For typos: don't just mirror the wrong word back. Say what the correct word means or why this spelling is wrong.
- For grammar: name the actual rule (e.g. "use past simple here", "missing article before a singular noun", "wrong preposition — use 'at' not 'in' with times").
- If the user mixed languages in one sentence, help them complete the whole thought in English.

Reply with ONLY a valid JSON object:

"annotated"
  Original text with HTML:
  • <s>word</s> — unnecessary or redundant word
  • <b>word</b> — wrong spelling, wrong form, or non-English word
  Plain text = correct.

"feedback"
  In English. Max 4 issues. For each:
  ❌ [what they wrote] → ✅ [correct version]
  One line only: the rule or reason. For typos just say "typo" — don't over-explain obvious misspellings.
  No "Explanation:" label. No "Corrections:" header. Just the list, clean and direct.
  If no issues: one short line starting "Looks clean —".

"improved"
  Full rewrite — natural fluent English, no mixed languages, same meaning.
  Label exactly: "Better version:"\
"""

TEST_SYSTEM_PROMPT = """\
Generate exactly 5 vocabulary quiz questions for a {level} level English learner.
Mix directions: some English→Russian, some Russian→English.
Use real, useful vocabulary appropriate for the level.
No repeating words.

Reply with ONLY a valid JSON array of 5 objects, each with:
  "question":      the question text, e.g. "What does 'ambitious' mean?" or "Как переводится 'уставший'?"
  "options":       array of exactly 5 strings (possible answers)
  "correct_index": integer 0–4 (index of the correct answer in options)
  "explanation":   one sentence explaining the correct answer in English\
"""

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "linda.db"


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id         INTEGER PRIMARY KEY,
                name            TEXT    NOT NULL DEFAULT '',
                level           TEXT    NOT NULL DEFAULT 'B1',
                onboarding_done INTEGER NOT NULL DEFAULT 0,
                history         TEXT    NOT NULL DEFAULT '[]',
                mistakes        TEXT    NOT NULL DEFAULT '',
                profile         TEXT    NOT NULL DEFAULT ''
            )
        """)
        con.commit()


def get_user(chat_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT name, level, onboarding_done, history, mistakes, profile FROM users WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
    if row:
        return {
            "name": row[0], "level": row[1], "onboarding_done": row[2],
            "history": json.loads(row[3]), "mistakes": row[4], "profile": row[5],
        }
    return {"name": "", "level": "B1", "onboarding_done": 0, "history": [], "mistakes": "", "profile": ""}


def save_user(chat_id: int, data: dict) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO users (chat_id, name, level, onboarding_done, history, mistakes, profile)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                name            = excluded.name,
                level           = excluded.level,
                onboarding_done = excluded.onboarding_done,
                history         = excluded.history,
                mistakes        = excluded.mistakes,
                profile         = excluded.profile
        """, (
            chat_id, data["name"], data["level"], data["onboarding_done"],
            json.dumps(data["history"], ensure_ascii=False),
            data["mistakes"], data.get("profile", ""),
        ))
        con.commit()


def history_last_24h(history: list) -> list:
    cutoff = time.time() - HOURS_24
    return [{"role": m["role"], "content": m["content"]}
            for m in history if m.get("ts", 0) >= cutoff]


# ── Typing indicator ──────────────────────────────────────────────────────────
async def keep_typing(bot: Bot, chat_id: int, stop: asyncio.Event, action: str = "typing") -> None:
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, action)
        except Exception:
            pass
        await asyncio.sleep(4)


# ── OpenAI ────────────────────────────────────────────────────────────────────
async def transcribe_audio(session: aiohttp.ClientSession, audio_bytes: bytes) -> str:
    data = aiohttp.FormData()
    data.add_field("model", STT_MODEL)
    data.add_field("file", audio_bytes, filename="voice.ogg", content_type="audio/ogg")
    async with session.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=OPENAI_HEADERS, data=data,
    ) as resp:
        resp.raise_for_status()
        return (await resp.json())["text"].strip()


async def gpt_json(session: aiohttp.ClientSession, messages: list, temp: float = 0.75) -> dict | list:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": temp,
    }
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={**OPENAI_HEADERS, "Content-Type": "application/json"},
        json=payload,
    ) as resp:
        resp.raise_for_status()
        raw = (await resp.json())["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
    # unwrap only if the TOP LEVEL itself is a list (test questions)
    # do NOT unwrap dicts that happen to contain lists as values
    if isinstance(parsed, list):
        return parsed
    return parsed


async def translate_to_russian(session: aiohttp.ClientSession, text: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "Translate the following text to Russian. Return only the translation."},
            {"role": "user",   "content": text},
        ],
        "temperature": 0.3,
    }
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={**OPENAI_HEADERS, "Content-Type": "application/json"},
        json=payload,
    ) as resp:
        resp.raise_for_status()
        return (await resp.json())["choices"][0]["message"]["content"].strip()


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────
async def synthesize_speech(session: aiohttp.ClientSession, text: str) -> bytes:
    payload = {
        "text": text,
        "model_id": EL_MODEL,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.80, "style": 0.15},
    }
    async with session.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE_ID}",
        headers=EL_HEADERS, json=payload,
    ) as resp:
        resp.raise_for_status()
        return await resp.read()


async def download_voice(bot: Bot, file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await bot.download_file(file.file_path, destination=tmp_path)
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


# ── Keyboards ─────────────────────────────────────────────────────────────────
def kb_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🎙 Speak",   callback_data="menu_chat"),
            InlineKeyboardButton(text="📝 Grammar", callback_data="menu_gram"),
            InlineKeyboardButton(text="🧪 Test",    callback_data="menu_test"),
        ]
    ])


def kb_after_user_voice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📝 Corrections", callback_data="explain"),
        InlineKeyboardButton(text="📊 My level",    callback_data="score"),
    ]])


def kb_bot_voice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📄 Show text", callback_data="showtext")],
        [InlineKeyboardButton(text="☰ Menu",       callback_data="menu")],
    ])


def kb_after_showtext() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Translate", callback_data="translate"),
        InlineKeyboardButton(text="☰ Menu",    callback_data="menu"),
    ]])


def kb_test_levels() -> InlineKeyboardMarkup:
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=lv, callback_data=f"test_level_{lv}") for lv in levels]
    ])


def kb_test_done() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="☰ Menu", callback_data="menu"),
    ]])


# ── Intro voice (onboarding) ──────────────────────────────────────────────────
async def handle_intro_voice(
    message: Message, bot: Bot,
    session: aiohttp.ClientSession, user_text: str,
) -> None:
    chat_id = message.chat.id
    parsed  = await gpt_json(session, [
        {"role": "system", "content": INTRO_SYSTEM_PROMPT},
        {"role": "user",   "content": f"User intro: {user_text}"},
    ], temp=0.8)

    spoken        = parsed.get("spoken", "")
    level         = parsed.get("level", "B1")
    mistakes_note = parsed.get("mistakes_note", "")
    name          = user_text.split()[0] if user_text else ""

    audio_bytes = await synthesize_speech(session, spoken)
    await message.answer_voice(
        BufferedInputFile(audio_bytes, filename="reply.mp3"),
        reply_markup=kb_bot_voice(),
    )
    pending[chat_id] = {"spoken": spoken, "feedback": "", "score": "", "last_text": spoken}

    ts = time.time()
    user_data = get_user(chat_id)
    user_data.update({
        "name": name, "level": level, "onboarding_done": 1,
        "mistakes": mistakes_note,
        "history": [
            {"role": "user",      "content": f"User intro: {user_text}", "ts": ts},
            {"role": "assistant", "content": spoken,                      "ts": ts},
        ],
    })
    save_user(chat_id, user_data)


# ── Main voice handler ────────────────────────────────────────────────────────
async def handle_voice_message(
    message: Message, bot: Bot,
    session: aiohttp.ClientSession, user_text: str,
) -> None:
    chat_id   = message.chat.id
    user_data = get_user(chat_id)

    user_content = f"Learner said: {user_text}"
    if user_data["mistakes"]:
        user_content += f"\n\nRecurring mistakes note: {user_data['mistakes']}"

    msgs = [{"role": "system", "content": MAIN_SYSTEM_PROMPT.format(
        level=user_data["level"],
        profile=user_data.get("profile") or "Nothing known yet.",
    )}]
    msgs.extend(history_last_24h(user_data["history"]))
    msgs.append({"role": "user", "content": user_content})

    parsed = await gpt_json(session, msgs)

    spoken           = parsed.get("spoken", "")
    annotated        = parsed.get("transcription_annotated", user_text)
    feedback         = parsed.get("feedback", "")
    score_assessment = parsed.get("score_assessment", "")
    mistakes_note    = parsed.get("mistakes_note", "")
    updated_level    = parsed.get("updated_level", user_data["level"])
    updated_profile  = parsed.get("updated_profile", user_data.get("profile", ""))

    await message.answer(
        f"<i>You said:</i>\n{annotated}",
        parse_mode="HTML",
        reply_markup=kb_after_user_voice(),
    )

    audio_bytes = await synthesize_speech(session, spoken)
    await message.answer_voice(
        BufferedInputFile(audio_bytes, filename="reply.mp3"),
        reply_markup=kb_bot_voice(),
    )

    pending[chat_id] = {
        "feedback": feedback, "score": score_assessment,
        "spoken": spoken, "last_text": spoken,
    }

    ts      = time.time()
    history = user_data["history"]
    history.append({"role": "user",      "content": f"Learner said: {user_text}", "ts": ts})
    history.append({"role": "assistant", "content": spoken,                        "ts": ts})
    user_data.update({
        "history": history, "mistakes": mistakes_note,
        "level": updated_level, "profile": updated_profile,
    })
    save_user(chat_id, user_data)


# ── Grammar check ─────────────────────────────────────────────────────────────
async def handle_grammar_check(
    message: Message, session: aiohttp.ClientSession, user_text: str,
) -> None:
    chat_id = message.chat.id
    parsed  = await gpt_json(session, [
        {"role": "system", "content": GRAMMAR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ])

    if not isinstance(parsed, dict):
        import re as _re
        if _re.search(r'[а-яёА-ЯЁ]', user_text):
            await message.answer(
                "Grammar check works with English text only.\n\n"
                "Send me your text in English and I'll check it.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="☰ Menu", callback_data="menu"),
                ]]),
            )
        else:
            await message.answer("Couldn't parse the response. Try again.")
        return

    annotated = parsed.get("annotated", user_text)
    feedback  = parsed.get("feedback", "")
    improved  = parsed.get("improved", "")

    parts = [f"<i>You wrote:</i>\n{annotated}"]
    if feedback:
        parts.append(feedback)
    if improved:
        parts.append(f"<i>{improved}</i>")

    full_text = f"{annotated}\n\n{improved}".strip()
    pending[chat_id] = pending.get(chat_id, {})
    pending[chat_id]["last_text"] = full_text

    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="☰ Menu", callback_data="menu"),
    ]])

    try:
        await message.answer("\n\n".join(parts), parse_mode="HTML", reply_markup=kb)
    except Exception as html_err:
        log.warning("HTML send failed (%s), falling back to plain text", html_err)
        # Strip HTML tags for plain text fallback
        import re
        plain_parts = [re.sub(r"<[^>]+>", "", p) for p in parts]
        await message.answer("\n\n".join(plain_parts), reply_markup=kb)


# ── Test: generate questions ──────────────────────────────────────────────────
async def generate_test_questions(session: aiohttp.ClientSession, level: str) -> list:
    result = await gpt_json(session, [
        {"role": "system", "content": TEST_SYSTEM_PROMPT.format(level=level)},
        {"role": "user",   "content": "Generate the quiz questions now."},
    ], temp=0.9)
    if isinstance(result, list):
        return result
    return []


async def send_test_question(bot: Bot, chat_id: int, question: dict, num: int, total: int) -> None:
    options = question.get("options", [])
    correct = question.get("correct_index", 0)
    explanation = question.get("explanation", "")

    poll = await bot.send_poll(
        chat_id=chat_id,
        question=f"{num}/{total}. {question['question']}",
        options=options,
        type="quiz",
        correct_option_id=correct,
        explanation=explanation,
        is_anonymous=False,
    )
    poll_registry[poll.poll.id] = {
        "chat_id": chat_id,
        "correct_index": correct,
        "question_num": num,
    }


# ── Handlers ──────────────────────────────────────────────────────────────────
def setup_handlers(dp: Dispatcher, bot: Bot, session: aiohttp.ClientSession) -> None:

    @dp.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        chat_id = message.chat.id
        user_data = get_user(chat_id)
        if user_data["onboarding_done"]:
            await message.answer(
                "Are you sure you want to start over? This will erase your entire chat history.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="Yes, reset", callback_data="confirm_reset"),
                    InlineKeyboardButton(text="No, keep it", callback_data="cancel_reset"),
                ]]),
            )
        else:
            gram_mode.discard(chat_id)
            test_sessions.pop(chat_id, None)
            await message.answer(GREETING, reply_markup=kb_menu())

    @dp.callback_query(F.data == "confirm_reset")
    async def cb_confirm_reset(call: CallbackQuery) -> None:
        await call.answer()
        chat_id = call.message.chat.id
        data = get_user(chat_id)
        data.update({"onboarding_done": 0, "history": [], "mistakes": "", "name": "", "level": "B1", "profile": ""})
        save_user(chat_id, data)
        gram_mode.discard(chat_id)
        test_sessions.pop(chat_id, None)
        await call.message.answer(GREETING, reply_markup=kb_menu())

    @dp.callback_query(F.data == "cancel_reset")
    async def cb_cancel_reset(call: CallbackQuery) -> None:
        await call.answer()
        await call.message.answer("Got it — nothing changed.", reply_markup=kb_menu())

    @dp.message(Command("menu"))
    async def cmd_menu(message: Message) -> None:
        await message.answer(MENU_TEXT, parse_mode="HTML", reply_markup=kb_menu())

    @dp.message(Command("gram"))
    async def cmd_gram(message: Message) -> None:
        gram_mode.add(message.chat.id)
        await message.answer(GRAM_PROMPT_MSG)

    @dp.message(Command("test"))
    async def cmd_test(message: Message) -> None:
        await message.answer("Choose your level:", reply_markup=kb_test_levels())

    @dp.message(Command("chat"))
    async def cmd_chat(message: Message) -> None:
        gram_mode.discard(message.chat.id)
        await message.answer("Ready! Send me a voice message.")

    @dp.message(F.voice)
    async def on_voice(message: Message) -> None:
        chat_id   = message.chat.id
        user_data = get_user(chat_id)
        gram_mode.discard(chat_id)
        stop = asyncio.Event()
        asyncio.create_task(keep_typing(bot, chat_id, stop, "record_voice"))
        try:
            audio_bytes = await download_voice(bot, message.voice.file_id)
            user_text   = await transcribe_audio(session, audio_bytes)
            log.info("[%s] transcribed: %s", chat_id, user_text)
            if not user_data["onboarding_done"] and chat_id in chat_mode:
                chat_mode.discard(chat_id)
                await handle_intro_voice(message, bot, session, user_text)
            elif not user_data["onboarding_done"]:
                await message.answer(
                    "Press 🎙 Speak in the menu first — I'll ask you a couple of questions.",
                    reply_markup=kb_menu(),
                )
            else:
                await handle_voice_message(message, bot, session, user_text)
        except aiohttp.ClientResponseError as e:
            log.error("API error: %s", e)
            await message.answer("API error. Try again in a moment.")
        except aiohttp.ClientResponseError as e:
            log.error("API error (voice): %s %s", e.status, e.message)
            await message.answer("API error. Try again in a moment.")
        except Exception as e:
            log.exception("Voice handler error: %s", e)
            await message.answer("Something went wrong. Try again.")
        finally:
            stop.set()

    @dp.message(F.text & ~F.text.startswith("/"))
    async def on_text(message: Message) -> None:
        chat_id   = message.chat.id
        user_data = get_user(chat_id)
        stop = asyncio.Event()
        asyncio.create_task(keep_typing(bot, chat_id, stop, "typing"))
        try:
            # Grammar always works — no onboarding required
            if chat_id in gram_mode or not user_data["onboarding_done"]:
                gram_mode.discard(chat_id)
                await handle_grammar_check(message, session, message.text)
            else:
                await handle_grammar_check(message, session, message.text)
        except aiohttp.ClientResponseError as e:
            log.error("API error (text): %s %s", e.status, e.message)
            await message.answer("API error. Try again.")
        except Exception as e:
            log.exception("Text handler error: %s", type(e).__name__, exc_info=e)
            await message.answer("Something went wrong. Try again.")
        finally:
            stop.set()

    # ── Poll answer tracking ──────────────────────────────────────────────────
    @dp.poll_answer()
    async def on_poll_answer(poll_answer: PollAnswer) -> None:
        poll_id = poll_answer.poll_id
        info    = poll_registry.get(poll_id)
        if not info:
            return

        chat_id = info["chat_id"]
        session_data = test_sessions.get(chat_id)
        if not session_data:
            return

        selected = poll_answer.option_ids[0] if poll_answer.option_ids else -1
        is_correct = (selected == info["correct_index"])
        if is_correct:
            session_data["correct_count"] += 1

        session_data["answered"] += 1
        answered = session_data["answered"]
        total    = len(session_data["questions"])

        if answered >= total:
            score  = session_data["correct_count"]
            emojis = {5: "🏆", 4: "💪", 3: "👍", 2: "😅", 1: "😬", 0: "💀"}
            emoji  = emojis.get(score, "🤔")
            level  = session_data["level"]
            await bot.send_message(
                chat_id,
                f"{emoji} <b>{score}/{total}</b> — {level} quiz done.\n\n"
                + (
                    "Clean sweep. Impressive." if score == 5 else
                    "Really solid, one or two slipped through." if score == 4 else
                    "Not bad. Room to grow." if score == 3 else
                    "Rough day. Try again." if score <= 2 else ""
                ),
                parse_mode="HTML",
                reply_markup=kb_test_done(),
            )
            test_sessions.pop(chat_id, None)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @dp.callback_query(F.data == "menu")
    async def cb_menu(call: CallbackQuery) -> None:
        await call.answer()
        await call.message.answer(MENU_TEXT, parse_mode="HTML", reply_markup=kb_menu())

    @dp.callback_query(F.data == "menu_chat")
    async def cb_menu_chat(call: CallbackQuery) -> None:
        await call.answer()
        chat_id   = call.message.chat.id
        gram_mode.discard(chat_id)
        user_data = get_user(chat_id)

        if not user_data["onboarding_done"]:
            chat_mode.add(chat_id)
            await call.message.answer(INTRO_PROMPT_MSG)
            return

        # Напоминаем последнюю тему если история есть
        history = history_last_24h(user_data["history"])
        if history:
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), None
            )
            if last_user:
                topic = last_user.replace("Learner said: ", "").strip()
                short = topic[:80] + ("…" if len(topic) > 80 else "")
                await call.message.answer(
                    f"Last time you were talking about:\n<i>«{short}»</i>\n\nSend a voice message to continue.",
                    parse_mode="HTML",
                )
                return

        await call.message.answer("Send me a voice message and let's talk.")

    @dp.callback_query(F.data == "menu_gram")
    async def cb_menu_gram(call: CallbackQuery) -> None:
        await call.answer()
        gram_mode.add(call.message.chat.id)
        await call.message.answer(GRAM_PROMPT_MSG)

    @dp.callback_query(F.data == "menu_test")
    async def cb_menu_test(call: CallbackQuery) -> None:
        await call.answer()
        await call.message.answer("Choose your level:", reply_markup=kb_test_levels())

    @dp.callback_query(F.data.startswith("test_level_"))
    async def cb_test_level(call: CallbackQuery) -> None:
        await call.answer()
        chat_id = call.message.chat.id
        level   = call.data.replace("test_level_", "")

        await call.message.answer(f"Generating your {level} quiz...")

        stop = asyncio.Event()
        asyncio.create_task(keep_typing(bot, chat_id, stop, "typing"))
        try:
            questions = await generate_test_questions(session, level)
            if not questions:
                await call.message.answer("Couldn't generate questions. Try again.")
                return

            test_sessions[chat_id] = {
                "level": level,
                "questions": questions,
                "answered": 0,
                "correct_count": 0,
            }

            for i, q in enumerate(questions, 1):
                await send_test_question(bot, chat_id, q, i, len(questions))
                await asyncio.sleep(0.3)
        except Exception as e:
            log.exception("Test error: %s", e)
            await call.message.answer("Something went wrong generating the quiz.")
        finally:
            stop.set()

    @dp.callback_query(F.data == "explain")
    async def cb_explain(call: CallbackQuery) -> None:
        await call.answer()
        text = pending.get(call.message.chat.id, {}).get("feedback") or "No data — send a new voice message."
        await call.message.answer(text, parse_mode="HTML")

    @dp.callback_query(F.data == "score")
    async def cb_score(call: CallbackQuery) -> None:
        await call.answer()
        text = pending.get(call.message.chat.id, {}).get("score") or "No data — send a new voice message."
        await call.message.answer(text)

    @dp.callback_query(F.data == "showtext")
    async def cb_showtext(call: CallbackQuery) -> None:
        await call.answer()
        text = pending.get(call.message.chat.id, {}).get("spoken") or "No data."
        pending[call.message.chat.id]["last_text"] = text
        await call.message.answer(
            f"<i>{text}</i>",
            parse_mode="HTML",
            reply_markup=kb_after_showtext(),
        )

    @dp.callback_query(F.data == "translate")
    async def cb_translate(call: CallbackQuery) -> None:
        await call.answer()
        chat_id = call.message.chat.id
        text    = pending.get(chat_id, {}).get("last_text") or "No text."
        stop    = asyncio.Event()
        asyncio.create_task(keep_typing(bot, chat_id, stop, "typing"))
        try:
            translated = await translate_to_russian(session, text)
            await call.message.answer(f"<i>{translated}</i>", parse_mode="HTML")
        finally:
            stop.set()


# ── Run ───────────────────────────────────────────────────────────────────────
async def main() -> None:
    init_db()
    bot = Bot(token=TELEGRAM_TOKEN)
    dp  = Dispatcher()
    async with aiohttp.ClientSession() as session:
        setup_handlers(dp, bot, session)
        log.info("Bot started")
        await dp.start_polling(bot, allowed_updates=["message", "callback_query", "poll_answer"])


if __name__ == "__main__":
    asyncio.run(main())
