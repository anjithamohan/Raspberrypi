import os

def announce(message):
    os.system(f'echo "{message}" | festival --tts')
