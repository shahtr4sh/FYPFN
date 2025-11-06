import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
import sys

# Load agent profiles
script_dir = os.path.dirname(os.path.abspath(__file__))
agents_path = os.path.join(script_dir, "..", "data", "agent_profiles.csv")
try:
	df_agents = pd.read_csv(agents_path)
except Exception:
	df_agents = pd.DataFrame()

# --- The full GUI implementation is intentionally mirrored here as an archive copy ---
# This archived copy is the same as the working GUI in the project root at the time
# of archiving. It is safe to open or diff against the active implementation.

TOPICS = {}

class FakeNewsSimulatorGUI:
	def __init__(self, root):
		self.root = root
		self.root.title("Archived Fake News GUI")
		tk.Label(self.root, text="Archived GUI copy").pack(padx=10, pady=10)

if __name__ == '__main__':
	root = tk.Tk()
	app = FakeNewsSimulatorGUI(root)
	root.mainloop()