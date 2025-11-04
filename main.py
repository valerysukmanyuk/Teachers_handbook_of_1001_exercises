import whisper
from agent import ExerciseAgent
import os
import getpass
# Teacher's_handbook_of_1001_exercises
# А вот это необязательно, но я гоняю на NPU
import intel_npu_acceleration_library as npu
from intel_npu_acceleration_library.compiler import CompilerConfig
import torch


def main():
    """Основная функция для запуска генерации упражнений из аудио/видео файлов"""
    # Загрузка и оптимизация модели Whisper
    print("Загрузка модели Whisper...")
    try:
        model = whisper.load_model("turbo")
    except Exception as e:
        print(f"Не удалось загрузить модель 'turbo': {e}")

    print("Пробубуем оптимизацию для Intel NPU...")
    try:
        compiler_conf = CompilerConfig(dtype=torch.int8, training=False)
        optimized_model = npu.compile(model, compiler_conf)
        whisper_model = optimized_model
    except Exception as e:
        print(f"Не удалось оптимизировать под NPU: {e}.")
        whisper_model = model

    if "DEEPSEEK_API_KEY" not in os.environ:
        print("Мы тут юзаем deepseek, поэтому гоните ключ")
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Введите ваш DEEPSEEK API ключ: ")

    # Создание агента и запуск
    print("Создание Агента для уровня C1...")
    agent = ExerciseAgent(whisper_model, level="C1")

    # Обработка файла
    for file_path in os.listdir("."):
        if file_path.endswith(('.mp4', '.mp3', '.wav', '.m4a', '.avi')):
            print(f"Обработка файла: {file_path}")
            try:
                result = agent.run(file_path)
                print("СГЕНЕРИРОВАННЫЕ УПРАЖНЕНИЯ:")
                print(result)
                
                # Сохранение результата
                output_file = "exercises_result.txt"
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{file_path} \n")
                    f.write(result)
                print(f"\nРезультат сохранен в: {output_file}")
                
            except Exception as e:
                print(f"Ошибка при обработке: {e}")

if __name__ == "__main__":
    main()