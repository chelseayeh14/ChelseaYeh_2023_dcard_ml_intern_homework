import pyautogui


def click_cloudflare(x=198, y=360):
    pyautogui.moveTo(x, y)
    pyautogui.click()


if __name__ == "__main__":
    while True:
        print(pyautogui.position())
