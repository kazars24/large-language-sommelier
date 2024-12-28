import os
import logging
import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TG_TOKEN')
API_BASE_URL = "http://localhost:8000"


class WineRecommendationBot:
    def __init__(self, token: str):
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.setup_handlers()

    def setup_handlers(self):
        """Установка обработчиков команд и сообщений"""
        # Команды
        self.dp.message.register(self.command_start, Command("start"))
        self.dp.message.register(self.command_help, Command("help"))
        
        # Текстовые сообщения
        self.dp.message.register(self.handle_message, F.text)

    async def get_wine_recommendation(self, question: str) -> str:
        """Получение рекомендации от API"""
        try:
            async with aiohttp.ClientSession() as session:  # Создаем новую сессию для каждого запроса
                async with session.post(
                    f"{API_BASE_URL}/recommend/",
                    json={"question": question},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)  # Добавляем таймаут
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["response"]
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: Status {response.status}, Response: {error_text}")
                        return f"Извините, произошла ошибка при получении рекомендации (Status: {response.status}). Попробуйте позже."
        except aiohttp.ClientError as e:
            logger.error(f"Network error in get_wine_recommendation: {e}")
            return "Извините, возникла проблема с подключением к сервису. Попробуйте позже."
        except Exception as e:
            logger.error(f"Unexpected error in get_wine_recommendation: {e}")
            return "Извините, произошла непредвиденная ошибка. Попробуйте позже."

    async def command_start(self, message: Message) -> None:
        """Обработчик команды /start"""
        welcome_message = (
            "👋 Здравствуйте! Я ваш персональный сомелье-бот.\n\n"
            "Я могу помочь вам с выбором вина. Просто задайте мне вопрос.\n\n"
            "Используйте /help для получения дополнительной информации."
        )
        await message.answer(welcome_message, parse_mode=ParseMode.MARKDOWN)

    async def command_help(self, message: Message) -> None:
        """Обработчик команды /help"""
        help_message = (
            "🍷 Я могу помочь вам выбрать подходящее вино. Вот что я умею:\n\n"
            "• Рекомендовать вина по вашим предпочтениям\n"
            "• Подбирать вина под определенные блюда\n"
            "• Рассказывать о характеристиках вин\n\n"
            "Просто напишите свой вопрос!"
        )
        await message.answer(help_message, parse_mode=ParseMode.MARKDOWN)

    async def handle_message(self, message: Message) -> None:
        """Обработчик текстовых сообщений"""
        try:
            # Отправка сообщения о начале обработки
            processing_msg = await message.answer("🤔 Обрабатываю ваш запрос...", parse_mode=ParseMode.MARKDOWN)

            # Получение рекомендации
            response = await self.get_wine_recommendation(message.text)

            try:
                # Попытка удалить сообщение о обработке
                await processing_msg.delete()
            except Exception as e:
                logger.error(f"Error deleting processing message: {e}")

            # Отправка ответа
            await message.answer(response, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in handle_message: {e}")
            await message.answer(
                "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже.",
                parse_mode=ParseMode.MARKDOWN
            )

    async def start(self):
        """Запуск бота"""
        try:
            logger.info("Starting bot...")
            await self.dp.start_polling(self.bot)
        finally:
            await self.bot.session.close()


async def check_api_availability() -> bool:
    """Проверка доступности API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/") as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"API check error: {e}")
        return False

async def main():
    # Проверка наличия токена
    if not TELEGRAM_TOKEN:
        logger.error("Telegram token not found in environment variables!")
        return

    # Проверка доступности API
    if not await check_api_availability():
        logger.error("Wine Recommendation API is not available!")
        return

    # Создание и запуск бота
    bot = WineRecommendationBot(TELEGRAM_TOKEN)
    await bot.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped!")