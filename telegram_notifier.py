# Уведомления в Telegram

import logging
import asyncio
from collections import deque
from typing import Optional, Callable
import html
import random
import re  
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramServerError, TelegramForbiddenError, TelegramUnauthorizedError

class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        max_retries: int = 20,
        retry_delay: float = 1.0,
        command_handler: Optional[Callable] = None  
    ):
        if not token or not chat_id:
            raise ValueError("Telegram token и chat_id должны быть указаны")
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger('telegram_notifier')
        self.message_queue = deque(maxlen=1000)
        self._lock = asyncio.Lock()
        self._queue_task = None
        self._rate_limiter = asyncio.Semaphore(1)

        # --- НОВОЕ: инициализация диспетчера для команд ---
        self.dp = Dispatcher()
        self.command_handler = command_handler
        if self.command_handler:
            self._setup_commands()

        self.logger.info("TelegramNotifier инициализирован")

    def _setup_commands(self):
        """Регистрирует обработчики команд Telegram"""
        @self.dp.message()
        async def handle_message(message: Message):
            if str(message.chat.id) != str(self.chat_id):
                self.logger.warning(f"Попытка команды от неавторизованного chat_id: {message.chat.id}")
                return
            if message.text and message.text.startswith('/'):
                await self.command_handler(message.text.strip(), message)

    async def start_polling(self):
        """Запуск опроса команд от Telegram (в фоне)"""
        if self.command_handler:
            self.logger.info("Запуск Telegram polling для команд...")
            await self.dp.start_polling(self.bot)

    async def start(self):
        if self._queue_task is None:
            self._queue_task = asyncio.create_task(self._process_queue())
            self.logger.info("Фоновая задача обработки очереди запущена")
        # Запускаем опрос команд тоже
        if self.command_handler:
            asyncio.create_task(self.start_polling())
    """ Экранирует специальные символы для корректной отправки в HTML-режиме. """
    def _escape_html(self, text: str) -> str:
        return html.escape(str(text))

    """ Асинхронная отправка сообщения в Telegram с HTML форматированием.		
    	Добавляет сообщение в очередь и возвращает True при успешной постановке в очередь. """
    async def send_message(self, message: str) -> bool:
        if not message:
            self.logger.warning("Попытка отправить пустое сообщение")
            return False
            # Экранирование сообщения для HTML
        safe_message = self._escape_html(message)
        async with self._lock:
            self.message_queue.append(safe_message)
            self.logger.debug(f"Сообщение добавлено в очередь: {safe_message[:50]}... (очередь: {len(self.message_queue)})")
        return True

    """ Обработка очереди сообщений в бесконечном цикле с повторными попытками. """
    async def _process_queue(self):
        while True:
            async with self._lock:
                if not self.message_queue:
                    await asyncio.sleep(1)
                    continue
                message = self.message_queue[0]
            success = await self._send_message_with_retry(message)
            async with self._lock:
                if success:
                    self.message_queue.popleft()
                    self.logger.info(f"Сообщение успешно отправлено: {message[:50]}...")
                else:
                    self.message_queue.popleft()
                    self.logger.error(f"Не удалось отправить сообщение после всех попыток, DROPPED: {message[:50]}...")

    async def _send_message_with_retry(self, message: str) -> bool:
        async with self._rate_limiter:
            for attempt in range(self.max_retries):
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=ParseMode.HTML
                    )
                    self.logger.debug(f"Сообщение отправлено с попытки {attempt + 1}: {message[:50]}...")
                    return True
                except TelegramBadRequest as e:
                    if "too many requests" in str(e).lower() or "flood control" in str(e).lower():
                        sleep_time = 30 + random.uniform(0, 1)
                        match = re.search(r'retry after (\d+)', str(e), re.IGNORECASE)
                        if match:
                            sleep_time = int(match.group(1)) + random.uniform(0, 1)
                        self.logger.warning(f"Ошибка Flood: {e}. Задержка {sleep_time:.2f} сек (попытка {attempt + 1})")
                        await asyncio.sleep(sleep_time)
                    else:
                        sleep_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Ошибка запроса: {e}. Задержка {sleep_time:.2f} сек (попытка {attempt + 1})")
                        await asyncio.sleep(sleep_time)
                except TelegramServerError as e:
                    sleep_time = self.retry_delay * (2 ** (attempt + 1)) + random.uniform(0, 1)
                    self.logger.warning(f"Ошибка сервера: {e}. Задержка {sleep_time:.2f} сек (попытка {attempt + 1})")
                    await asyncio.sleep(sleep_time)
                except (TelegramForbiddenError, TelegramUnauthorizedError) as e:
                    self.logger.error(f"Критическая ошибка авторизации: {e}.")
                    return False
                except Exception as e:
                    sleep_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(f"Неизвестная ошибка: {e}. Задержка {sleep_time:.2f} сек (попытка {attempt + 1})")
                    await asyncio.sleep(sleep_time)
            return False
    
    """ Закрытие сессии бота для graceful shutdown """
    async def close(self):
        if self._queue_task:
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                self.logger.info("Фоновая задача обработки очереди остановлена")
        await self.bot.session.close()
        self.logger.info("Сессия Telegram бота закрыта")