"""Main entry point for the Fake News Simulator application."""

import tkinter as tk
import sys
from gui import FakeNewsSimulatorGUI

def main():
    root = tk.Tk()
    app = FakeNewsSimulatorGUI(root)

    def on_main_close():
        root.quit()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_main_close)

    root.mainloop()
    sys.exit()

if __name__ == "__main__":
    main()
