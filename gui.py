"""GUI implementation for the Fake News Simulator."""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os
import json
import glob

from simulator import FakeNewsSimulator
from pbm_simulator import PopulationSimulator
from analysis import analyze_context_juiciness, infer_topic_from_context
from config import TOPICS, TOPIC_CATEGORIES


# Simple tooltip helper for Tkinter widgets
class ToolTip:
    """Create a tooltip for a given widget.

    Usage:
        tt = ToolTip(widget, text='helpful text')
        # update text later: tt.text = 'new text'
    """
    def __init__(self, widget, text=''):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, event=None):
        if not self.text:
            return
        if self.tipwindow or not self.text:
            return
        x = y = 0
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4, ipady=2)

    def hide(self, event=None):
        tw = self.tipwindow
        if tw:
            tw.destroy()
        self.tipwindow = None

class FakeNewsSimulatorGUI:
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Fake News Circulation Simulator")
        # Start the GUI maximized/full-screen for better visibility
        try:
            # Windows: maximize window
            self.root.state('zoomed')
        except Exception:
            try:
                # Fallback to fullscreen if state('zoomed') not supported
                self.root.attributes('-fullscreen', True)
            except Exception:
                pass
        
        # Initialize variables
        self.topic = tk.StringVar(value="Ransomware Alert")
        self.juiciness = tk.IntVar(value=50)
        # Agent sources are fixed: ABM uses CSV, PBM uses random-average agents
        self.context_text = tk.StringVar(value="")
        
        # Simulation state
        self.round = 0
        self.max_rounds = 24  # default to 24 rounds (1 day / 24 hours)
        self.intervention = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []
        self.simulator = None
        
        # Load agents data
        # ABM CSV chooser (default will be chosen in setup_ui)
        self.abm_csv_var = tk.StringVar()
        self._load_agent_data()
        
        # Set up the UI
        self.setup_ui()

    def _load_agent_data(self):
        """Load agent profiles from CSV."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        # Discover CSV files in data directory
        csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
        # Choose default file (prefer any file containing 'agent_profiles')
        default_file = None
        for f in csv_files:
            if 'agent_profiles' in f.lower():
                default_file = f
                break
        if default_file is None and csv_files:
            default_file = csv_files[0]

        # If abm_csv_var not yet set, set to default
        if not self.abm_csv_var.get():
            self.abm_csv_var.set(default_file or '')

        selected = self.abm_csv_var.get()
        agents_path = os.path.join(data_dir, selected) if selected else os.path.join(data_dir, "agent_profiles1.csv")
        self.df_agents = pd.read_csv(agents_path)

    # --- Scenario management helpers ---------------------------------
    def _ensure_scenarios_dir(self):
        """Ensure a local `scenarios/` directory exists for JSON scenario files."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.scenarios_dir = os.path.join(script_dir, 'scenarios')
        try:
            os.makedirs(self.scenarios_dir, exist_ok=True)
        except Exception:
            pass

    def _load_scenarios_list(self):
        """Scan the scenarios directory and populate the combobox values."""
        try:
            pattern = os.path.join(self.scenarios_dir, '*.json')
            files = glob.glob(pattern)
            names = [os.path.basename(f) for f in sorted(files)]
            self.scenario_combo['values'] = names
        except Exception:
            self.scenario_combo['values'] = []

    def _on_scenario_load(self):
        """Load selected scenario JSON and apply parameters to the UI."""
        sel = self.scenario_var.get()
        if not sel:
            messagebox.showinfo('Load Scenario', 'No scenario selected')
            return
        path = os.path.join(getattr(self, 'scenarios_dir', ''), sel)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            self._apply_scenario(data)
            messagebox.showinfo('Load Scenario', f'Applied scenario: {data.get("name", sel)}')
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def _apply_scenario(self, data: dict):
        """Apply scenario dictionary to GUI controls."""
        try:
            # Basic fields
            if 'context' in data:
                self.context_text.set(data.get('context', ''))
            if 'abm_csv' in data:
                # set csv combobox if file exists under data/
                script_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(script_dir, 'data', data.get('abm_csv'))
                if os.path.exists(candidate):
                    self.abm_csv_var.set(data.get('abm_csv'))
                    self._load_agent_data()
            if 'juiciness' in data:
                try:
                    self.juiciness.set(int(data.get('juiciness')))
                except Exception:
                    pass
            if 'max_rounds' in data:
                try:
                    self.max_rounds_var.set(int(data.get('max_rounds')))
                except Exception:
                    pass
            if 'intervention_round' in data:
                try:
                    self.intervention_round_var.set(int(data.get('intervention_round')))
                except Exception:
                    pass

            # ABM sharing params mapping
            abm_params = data.get('abm_sharing_params', {}) or {}
            # support alternate key names
            def _set_if(varname, dkeys):
                for k in dkeys:
                    if k in abm_params:
                        try:
                            getattr(self, varname).set(float(abm_params[k]))
                        except Exception:
                            pass
                        break

            _set_if('share_belief_var', ['share_belief_weight', 'share_belief_var'])
            _set_if('share_emotion_var', ['share_emotional_weight', 'share_emotion_var'])
            _set_if('share_conf_var', ['share_conf_weight', 'share_confirmation_weight', 'share_conf_var'])
            _set_if('share_juice_var', ['share_juice_weight', 'share_juice_var'])
            _set_if('share_crit_var', ['share_crit_var', 'share_critical_penalty', 'share_crit'])
            _set_if('share_offset_var', ['share_offset', 'share_offset_var'])
            if 'attribution_mode' in abm_params:
                # set human-label combo directly if present
                try:
                    self.attribution_var.set(abm_params.get('attribution_mode'))
                except Exception:
                    pass

            # PBM rates
            pbm = data.get('pbm_rates', {}) or {}
            try:
                if 'contact_rate' in pbm:
                    self.pbm_contact_rate_var.set(float(pbm.get('contact_rate')))
                if 'belief_rate' in pbm:
                    self.pbm_belief_rate_var.set(float(pbm.get('belief_rate')))
                if 'recovery_rate' in pbm:
                    self.pbm_recovery_rate_var.set(float(pbm.get('recovery_rate')))
                if 'initial_believers' in pbm:
                    self.pbm_initial_believers_var.set(int(pbm.get('initial_believers')))
            except Exception:
                pass

            # advanced params
            adv = data.get('advanced_params', {}) or {}
            try:
                if 'calibration_multiplier' in adv:
                    self.calibration_multiplier_var.set(float(adv.get('calibration_multiplier')))
            except Exception:
                pass
        except Exception:
            # best-effort apply
            pass

    def _save_current_scenario(self):
        """Save the current GUI parameters into a scenario JSON file."""
        try:
            # Ensure scenario dir
            self._ensure_scenarios_dir()
            from tkinter import filedialog
            # Suggest file name from topic or timestamp
            suggested = (self._sim_topic or 'scenario').replace(' ', '_') + '.json'
            fname = filedialog.asksaveasfilename(initialdir=self.scenarios_dir, initialfile=suggested, defaultextension='.json', filetypes=[('JSON files','*.json')])
            if not fname:
                return
            scenario = {
                'name': os.path.splitext(os.path.basename(fname))[0],
                'description': '',
                'context': self.context_text.get(),
                'abm_csv': self.abm_csv_var.get(),
                'juiciness': int(self.juiciness.get()),
                'max_rounds': int(self.max_rounds_var.get()),
                'intervention_round': int(self.intervention_round_var.get()),
                'abm_sharing_params': {
                    'share_belief_weight': float(self.share_belief_var.get()),
                    'share_emotional_weight': float(self.share_emotion_var.get()),
                    'share_conf_weight': float(self.share_conf_var.get()),
                    'share_juice_weight': float(self.share_juice_var.get()),
                    'attribution_mode': self.attribution_var.get()
                },
                'pbm_rates': {
                    'contact_rate': float(self.pbm_contact_rate_var.get()),
                    'belief_rate': float(self.pbm_belief_rate_var.get()),
                    'recovery_rate': float(self.pbm_recovery_rate_var.get()),
                    'initial_believers': int(self.pbm_initial_believers_var.get())
                },
                'advanced_params': {
                    'calibration_multiplier': float(self.calibration_multiplier_var.get())
                },
                'events': []
            }
            with open(fname, 'w', encoding='utf-8') as fh:
                json.dump(scenario, fh, indent=2)
            # refresh list
            self._load_scenarios_list()
            messagebox.showinfo('Saved', f'Saved scenario to {fname}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def setup_ui(self):
        """Set up the user interface with a modern, streamlined layout.

        Layout design:
        - Left rail: scrollable parameters and file chooser (compact, grouped)
        - Center: tabbed area (Visualize, Summary)
        - Right: quick controls / status
        """
        # Global styles and fonts
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        note_font = ('Segoe UI', 9, 'bold')
        default_font = ('Segoe UI', 10)
        heading_font = ('Segoe UI', 11, 'bold')

        # Main paned layout
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Left: parameters (scrollable) ---
        left_container = ttk.Frame(main_pane, width=320)
        main_pane.add(left_container, weight=0)

        # Scrollable parameters area
        params_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0)
        params_scroll = ttk.Scrollbar(left_container, orient='vertical', command=params_canvas.yview)
        params_frame = ttk.Frame(params_canvas)
        params_frame.bind('<Configure>', lambda e: params_canvas.configure(scrollregion=params_canvas.bbox('all')))
        params_canvas.create_window((0,0), window=params_frame, anchor='nw')
        params_canvas.configure(yscrollcommand=params_scroll.set)
        params_canvas.pack(side='left', fill='both', expand=True)
        params_scroll.pack(side='right', fill='y')

        # Header and context
        ttk.Label(params_frame, text='Simulation Parameters', font=heading_font).pack(anchor='w', pady=(6,4), padx=8)
        ttk.Label(params_frame, text='Fake News Context:', font=default_font).pack(anchor='w', padx=8)
        self.context_entry = ttk.Entry(params_frame, textvariable=self.context_text, width=36)
        self.context_entry.pack(anchor='w', padx=8, pady=(0,6))

        # ABM CSV chooser
        ttk.Label(params_frame, text='ABM agent CSV:', font=default_font).pack(anchor='w', padx=8)
        # populate CSV list
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
        self.abm_csv_var.set(self.abm_csv_var.get() or (csv_files[0] if csv_files else ''))
        self.abm_csv_combo = ttk.Combobox(params_frame, values=csv_files, textvariable=self.abm_csv_var, state='readonly', width=34)
        self.abm_csv_combo.pack(anchor='w', padx=8, pady=(0,6))
        self.abm_csv_combo.bind('<<ComboboxSelected>>', lambda e: self._load_agent_data())

        # Scenario picker: load/save predefined scenarios from ./scenarios/*.json
        ttk.Label(params_frame, text='Scenario:', font=default_font).pack(anchor='w', padx=8)
        self.scenario_var = tk.StringVar()
        # populate later after we ensure the scenarios directory exists
        self.scenario_combo = ttk.Combobox(params_frame, values=[], textvariable=self.scenario_var, state='readonly', width=34)
        self.scenario_combo.pack(anchor='w', padx=8, pady=(0,6))
        btn_frame_scn = ttk.Frame(params_frame)
        btn_frame_scn.pack(fill='x', padx=8, pady=(0,8))
        ttk.Button(btn_frame_scn, text='Load Scenario', command=self._on_scenario_load).pack(side='left', expand=True, fill='x', padx=(0,6))
        ttk.Button(btn_frame_scn, text='Save Current as Scenario', command=self._save_current_scenario).pack(side='left', expand=True, fill='x')

        # Juiciness slider (ttk Scale not available cross-platform, keep tk.Scale but styled)
        ttk.Label(params_frame, text='Juiciness:', font=default_font).pack(anchor='w', padx=8)
        self.juiciness_scale = tk.Scale(params_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.juiciness, length=260)
        self.juiciness_scale.pack(anchor='w', padx=8, pady=(0,8))

        ttk.Separator(params_frame, orient='horizontal').pack(fill='x', pady=6, padx=6)

        # Group: PBM params
        ttk.Label(params_frame, text='Population Model (PBM)', font=heading_font).pack(anchor='w', padx=8, pady=(6,4))
        def add_scale(label, var, low, high, step=0.01):
            ttk.Label(params_frame, text=label, font=default_font).pack(anchor='w', padx=8)
            s = tk.Scale(params_frame, from_=low, to=high, resolution=step, orient=tk.HORIZONTAL, variable=var, length=260)
            s.pack(anchor='w', padx=8, pady=(0,8))
        self.pbm_contact_rate_var = tk.DoubleVar(value=0.4)
        add_scale('PBM contact rate', self.pbm_contact_rate_var, 0.0, 1.0)
        self.pbm_belief_rate_var = tk.DoubleVar(value=0.3)
        add_scale('PBM belief rate', self.pbm_belief_rate_var, 0.0, 1.0)
        self.pbm_recovery_rate_var = tk.DoubleVar(value=0.1)
        add_scale('PBM recovery rate', self.pbm_recovery_rate_var, 0.0, 1.0)
        ttk.Label(params_frame, text='PBM initial believers:', font=default_font).pack(anchor='w', padx=8)
        self.pbm_initial_believers_var = tk.IntVar(value=5)
        tk.Spinbox(params_frame, from_=0, to=10000, textvariable=self.pbm_initial_believers_var, width=12).pack(anchor='w', padx=8, pady=(0,8))

        ttk.Separator(params_frame, orient='horizontal').pack(fill='x', pady=6, padx=6)

        # Group: ABM sharing model controls
        ttk.Label(params_frame, text='Agent-Based Model (ABM)', font=heading_font).pack(anchor='w', padx=8, pady=(6,4))
        ttk.Label(params_frame, text='*Higher weights indicate more influence.', font=note_font).pack(anchor='w', padx=8, pady=(1,1))
        self.share_belief_var = tk.DoubleVar(value=4.0)
        add_scale('Belief Weight', self.share_belief_var, 0.0, 8.0, 0.1)
        self.share_emotion_var = tk.DoubleVar(value=1.5)
        add_scale('Emotional Weight', self.share_emotion_var, 0.0, 4.0, 0.1)
        self.share_conf_var = tk.DoubleVar(value=1.0)
        add_scale('Confirmation Bias Weight', self.share_conf_var, 0.0, 4.0, 0.1)
        self.share_juice_var = tk.DoubleVar(value=1.0)
        add_scale('Virality Multiplier', self.share_juice_var, 0.0, 3.0, 0.05)

        ttk.Separator(params_frame, orient='horizontal').pack(fill='x', pady=6, padx=6)

        # Attribution mode
        ttk.Label(params_frame, text='Transmission attribution:', font=default_font).pack(anchor='w', padx=8)
        self.attribution_var = tk.StringVar(value='Weighted by source belief')
        attr_options = ['Weighted by source belief', 'Random among sharers', 'First sharer']
        self.attribution_combo = ttk.Combobox(params_frame, values=attr_options, textvariable=self.attribution_var, state='readonly', width=30)
        self.attribution_combo.pack(anchor='w', padx=8, pady=(0,8))

        # Calibration and run controls at bottom of params rail
        ttk.Separator(params_frame, orient='horizontal').pack(fill='x', pady=6, padx=6)
        ttk.Label(params_frame, text='Advanced', font=heading_font).pack(anchor='w', padx=8, pady=(6,4))
        self.calibration_multiplier_var = tk.DoubleVar(value=3.0)
        add_scale('Calibration multiplier', self.calibration_multiplier_var, 0.0, 10.0, 0.1)
        ttk.Label(params_frame, text='Intervention round:', font=default_font).pack(anchor='w', padx=8)
        self.intervention_round_var = tk.IntVar(value=5)
        tk.Spinbox(params_frame, from_=0, to=1000, textvariable=self.intervention_round_var, width=12).pack(anchor='w', padx=8, pady=(0,8))
        ttk.Label(params_frame, text='Max rounds:', font=default_font).pack(anchor='w', padx=8)
        self.max_rounds_var = tk.IntVar(value=self.max_rounds)
        tk.Spinbox(params_frame, from_=1, to=1000, textvariable=self.max_rounds_var, width=12).pack(anchor='w', padx=8, pady=(0,12))

        # Run / Reset buttons (compact)
        btns = ttk.Frame(params_frame)
        btns.pack(fill='x', padx=8, pady=(6,12))
        self.run_btn = ttk.Button(btns, text='Run Simulation', command=self.init_simulation)
        self.run_btn.pack(side='left', expand=True, fill='x', padx=(0,6))
        self.reset_btn = ttk.Button(btns, text='Reset', command=self.reset_simulation)
        self.reset_btn.pack(side='left', expand=True, fill='x')
        self.reset_btn.state(['disabled'])

        # --- Center: tabbed visualization and summary ---
        center = ttk.Frame(main_pane)
        main_pane.add(center, weight=1)

        self.notebook = ttk.Notebook(center)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=8)

        # Visualize tab (holds matplotlib canvas)
        self.tab_visualize = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_visualize, text='Visualize')

        # Summary tab (populated after run)
        self.tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text='Summary')

        # Comparison tab (ABM vs PBM)
        self.tab_comparison = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_comparison, text='Comparison')

        # Right: quick status and helpful actions
        right = ttk.Frame(main_pane, width=220)
        main_pane.add(right, weight=0)
        ttk.Label(right, text='Status', font=heading_font).pack(anchor='w', padx=8, pady=(8,4))
        self.round_label = ttk.Label(right, text=f'Round: {self.round}', font=default_font)
        self.round_label.pack(anchor='w', padx=8)
        self.status_label = ttk.Label(right, text='Ready', font=default_font)
        self.status_label.pack(anchor='w', padx=8, pady=(4,8))

        ttk.Separator(right, orient='horizontal').pack(fill='x', padx=8, pady=6)
        ttk.Label(right, text='Quick Actions', font=heading_font).pack(anchor='w', padx=8, pady=(4,6))
        ttk.Button(right, text='Export Transmission Log', command=self._export_transmission_log).pack(fill='x', padx=8, pady=4)
        ttk.Button(right, text='Export Spreaders', command=self._export_spreaders).pack(fill='x', padx=8, pady=4)

        # --- Matplotlib canvas in Visualize tab ---
        self.fig = plt.figure(figsize=(10,6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_visualize)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)
        self.fig.suptitle('Fake News Spread - Round 0', y=0.95, fontsize=12)
        self.canvas.draw()

        # small binding: when user selects Summary tab, populate it
        def _on_tab_changed(event):
            selected = event.widget.select()
            if event.widget.tab(selected, 'text') == 'Summary':
                self._populate_summary_tab()
        self.notebook.bind('<<NotebookTabChanged>>', _on_tab_changed)

        # Populate scenarios list from disk
        try:
            self._ensure_scenarios_dir()
            self._load_scenarios_list()
        except Exception:
            pass

    def init_simulation(self):
        """Initialize a new simulation."""
        # Analyze context and set juiciness
        context = self.context_text.get().lower()
        juiciness_score = analyze_context_juiciness(context)
        self.juiciness.set(juiciness_score)
        self.juiciness_scale.set(juiciness_score)

        # Infer topic
        topic, topic_weight, topic_category = infer_topic_from_context(
            context, TOPICS, TOPIC_CATEGORIES
        )
        self._sim_topic = topic
        self._sim_topic_weight = topic_weight
        self._sim_topic_category = topic_category
        
        # Reset simulation state
        self.round = 0
        self.round_label.config(text=f"Round: {self.round}")
        self.intervention = False
        self.extra_rounds = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []
        # Apply max rounds override from parameters panel (if set)
        try:
            self.max_rounds = int(self.max_rounds_var.get())
        except Exception:
            pass
        
        # Initialize simulators and start simulation
        self._init_simulators()

        # Update UI: enable reset, disable run
        try:
            self.reset_btn.state(['!disabled'])
            self.run_btn.state(['disabled'])
        except Exception:
            pass
        self.update_graph()

    def automate_rounds(self):
        """Automatically run simulation rounds."""
        if self.round >= self.max_rounds:
            return
        
        # Apply intervention at configured round
        int_round = int(self.intervention_round_var.get()) if hasattr(self, 'intervention_round_var') else 5
        if self.round == int_round and not self.intervention:
            self.intervention = True
            self.intervention_rounds.append(self.round)
            
        # Run the next round
        self.run_next_round()
        
        # Schedule the next round after a delay
        if self.round < self.max_rounds:  # Double check we haven't hit max rounds
            self.root.after(500, self.automate_rounds)  # Increased delay to 1 second for better visualization

    def run_next_round(self):
        """Run the next simulation round for both models."""
        if self.round >= self.max_rounds:
            return

        # Run ABM simulation step
        if self._sim_topic == "Financial Scam":
            abm_result = self.abm_simulator.simulate_scam_round()
            self.scam_history.append(abm_result)
            self.abm_results['believer_counts'].append(sum(1 for x in abm_result if x))
        else:
            abm_result = self.abm_simulator.simulate_fake_news_round(
                self.juiciness.get() / 100.0,
                self._sim_topic_weight,
                self._sim_topic_category,
                self.intervention,
                getattr(self, 'extra_rounds', False)
            )
            self.round_history.append(abm_result)
            self.abm_results['believer_counts'].append(sum(1 for x in abm_result if x))
       
        # Run PBM simulation step and update rates if needed
        if self.intervention:
            self.pbm_simulator.adjust_rates(
                self._sim_topic_weight,
                self.juiciness.get() / 100.0,
                True
            )
        susceptible, believers, immune = self.pbm_simulator.simulate_step()
        self.pbm_results['susceptible'].append(susceptible)
        self.pbm_results['believers'].append(believers)
        self.pbm_results['immune'].append(immune)

        # Update round counter and UI
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}")
        
        # Update visualization
        self.update_graph()
        
        # Show summary if we've reached max rounds
        if self.round >= self.max_rounds:
            self.show_summary()

    def update_graph(self, highlight_ids=None):
        # Clear the entire figure
        self.fig.clear()
        
        # Create a new gridspec with proper spacing
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        
        # Set the main title with proper spacing
        self.fig.suptitle(f"Fake News Spread - Round {self.round}", y=0.95, fontsize=12)
        
        # Create subplots with the gridspec
        ax1 = self.fig.add_subplot(gs[0])  # ABM subplot
        ax2 = self.fig.add_subplot(gs[1])  # PBM subplot
        
        # Setup colors and patches for legends
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Believers')
        blue_patch = mpatches.Patch(color='blue', label='Non-believers')
        
        # ABM Network visualization (left)
        is_scam = self._sim_topic == "Financial Scam"
        colors = self.abm_simulator.get_node_colors(is_scam)
        
        # Calculate node positions (layout) - use a more expansive spring layout
        # to spread nodes further apart for readability. Recompute on each
        # visualization update to ensure the layout uses the chosen spacing.
        try:
            # Aggressively increase the optimal distance (k) and scale so nodes
            # are placed much further apart. Increase iterations for stability.
            self.pos = nx.spring_layout(
                self.abm_simulator.G,
                k=4.0,          # desired distance between nodes (larger = farther apart)
                iterations=500,
                scale=3.0,
                seed=42
            )
        except Exception:
            # Fallback to a simple layout on error
            self.pos = nx.spring_layout(self.abm_simulator.G, seed=42)
            
        # Draw the ABM network with enhanced styling
        nx.draw_networkx_edges(self.abm_simulator.G,
                               pos=self.pos,
                               edge_color='gray',
                               width=1.0,
                               alpha=0.5,
                               ax=ax1)

        # Allow optional highlighting of specific agent node IDs (e.g. when the
        # user double-clicks a spreader in the Summary tab). Map node ids to the
        # order used by NetworkX (list(self.abm_simulator.G.nodes())).
        try:
            base_colors = list(colors)
            if highlight_ids:
                nodes_order = list(self.abm_simulator.G.nodes())
                # build a set of ints for quick test
                hset = set()
                for hid in highlight_ids:
                    try:
                        hset.add(int(hid))
                    except Exception:
                        pass
                for hid in hset:
                    if hid in nodes_order:
                        idx = nodes_order.index(hid)
                        # use a bright highlight color
                        base_colors[idx] = 'yellow'
            draw_colors = base_colors
        except Exception:
            draw_colors = colors

        # Draw nodes with better visibility (reduced size to help label legibility)
        nx.draw_networkx_nodes(self.abm_simulator.G,
                               pos=self.pos,
                               node_color=draw_colors,
                               node_size=300,  # smaller node size to reduce label overlap
                               edgecolors='black',  # Black border for nodes
                               linewidths=1.0,
                               alpha=0.95,
                               ax=ax1)

        # Add labels with better visibility
        nx.draw_networkx_labels(self.abm_simulator.G,
                                pos=self.pos,
                                font_size=7,
                                font_weight='normal',
                                font_color='black',
                                ax=ax1)
        
        # Add legend and border for ABM plot
        ax1.legend(handles=[red_patch, blue_patch],
                  loc='upper right',
                  title="Agent States",
                  title_fontsize=10,
                  fontsize=9,
                  framealpha=0.9,
                  edgecolor='black')
        
        # Set title and style plot
        ax1.set_title("Agent-Based Model (ABM)", pad=10, fontsize=12, fontweight='bold')
        ax1.set_facecolor('#f8f8f8')  # Light gray background
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
            
        # Ensure proper aspect ratio
        ax1.set_aspect('equal')

        # Fixed border styling (thicker border for a corporate look)
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)  # Thicker border
            spine.set_color('black')  # Darker border color

        # Compute dynamic axis limits from node positions so the border
        # has ample padding and all nodes fit evenly inside the box.
        try:
            if hasattr(self, 'pos') and self.pos:
                xs = np.array([p[0] for p in self.pos.values()])
                ys = np.array([p[1] for p in self.pos.values()])
                x_min, x_max = float(xs.min()), float(xs.max())
                y_min, y_max = float(ys.min()), float(ys.max())
                x_span = max(1e-6, x_max - x_min)
                y_span = max(1e-6, y_max - y_min)
                # Use the larger span to compute a symmetric padding so the
                # network sits comfortably centered in the box.
                # Increase padding multiplier so nodes have more breathing room
                # and labels are less likely to overlap the border.
                pad = max(x_span, y_span) * 1.2
                ax1.set_xlim(x_min - pad, x_max + pad)
                ax1.set_ylim(y_min - pad, y_max + pad)
            else:
                # Fallback fixed limits
                ax1.set_xlim(-1.5, 1.5)
                ax1.set_ylim(-1.5, 1.5)
        except Exception:
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
        
        # PBM visualization (right)
        if hasattr(self, 'pbm_simulator'):
            # Get total population and current data
            total_population = len(self.abm_simulator.G.nodes())
            pbm_history = self.pbm_simulator.get_history()
            
            # Get number of rounds to plot
            current_round = self.round + 1
            rounds = range(current_round)
            
            # Plot PBM data
            pbm_believers = pbm_history['believers'][:current_round]
            if pbm_believers:  # Only plot if we have data
                # Plot susceptible population (blue dashed)
                pbm_susceptible = pbm_history['susceptible'][:current_round]
                ax2.plot(rounds, pbm_susceptible, 'b--', label='Susceptible', linewidth=2)
                ax2.scatter(rounds, pbm_susceptible, color='blue', s=50, alpha=0.6, marker='o')
                
                # Plot believer population (red solid)
                ax2.plot(rounds, pbm_believers, 'r-', label='Believers', linewidth=2)
                ax2.scatter(rounds, pbm_believers, color='red', s=50, alpha=0.6, marker='o')
                
                # Set axis limits for better visualization
                ax2.set_ylim(0, total_population * 1.1)
                ax2.set_xlim(-0.5, max(9.5, current_round - 0.5))
                
                # Add grid and labels
                ax2.grid(True, alpha=0.2, linestyle='--')
                ax2.set_xlabel('Round', fontsize=9)
                ax2.set_ylabel('Population', fontsize=9)
                ax2.set_title('Population-Based Model (PBM)', pad=10, fontsize=10, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
                
                # Add spines
                for spine in ax2.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
        
        # Add titles
        ax1.set_title("Agent-Based Model (ABM)", pad=10, fontsize=10, fontweight='bold')
        ax2.set_title("Population-Based Model (PBM)", pad=10, fontsize=10, fontweight='bold')
        
        # Add borders to both plots
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color('black')
        
        # Add legend to ABM plot
        ax1.legend(handles=[red_patch, blue_patch],
                  loc='upper right',
                  fontsize=8,
                  framealpha=0.9)
        
        # Draw the updated canvas
        self.canvas.draw()

    def _export_transmission_log(self):
        """Export ABM transmission history to CSV (quick action)."""
        try:
            from tkinter import filedialog
            fname = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv')])
            if not fname:
                return
            with open(fname, 'w', newline='') as fh:
                fh.write('round,source,target\n')
                for r, trans in enumerate(getattr(self.abm_simulator, 'transmission_history', []), start=1):
                    for (s, t) in trans:
                        fh.write(f"{r},{s},{t}\n")
            messagebox.showinfo('Export', f'Wrote transmissions to {fname}')
        except Exception as e:
            messagebox.showerror('Export error', str(e))

    def _export_spreaders(self):
        """Export the list of unique spreader agent IDs to a CSV file."""
        try:
            from tkinter import filedialog
            fname = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv')])
            if not fname:
                return
            # Compute unique spreaders from round_history
            unique = set()
            for rnd in getattr(self, 'round_history', []):
                try:
                    ids = [i for i, v in enumerate(rnd) if v]
                    unique.update(ids)
                except Exception:
                    pass
            with open(fname, 'w', newline='') as fh:
                fh.write('agent_id\n')
                for aid in sorted(unique):
                    fh.write(f"{aid}\n")
            messagebox.showinfo('Export', f'Wrote {len(unique)} spreader IDs to {fname}')
        except Exception as e:
            messagebox.showerror('Export error', str(e))

    def _populate_summary_tab(self):
        """Populate the Summary tab in the central notebook with latest results."""
        # Clear any existing content and use the Summary tab directly (no scrollable wrapper)
        for w in self.tab_summary.winfo_children():
            w.destroy()
        content = self.tab_summary

        # Summary header
        header = ttk.Label(content, text='Simulation Summary', font=('Segoe UI', 12, 'bold'))
        header.pack(anchor='w', padx=8, pady=(8,8))

        # Compute ABM stats
        victim_counts = [sum(1 for a in round_data if a) for round_data in self.round_history] if hasattr(self, 'round_history') else []
        total_shares = sum(victim_counts) if victim_counts else 0
        peak_believers = max(victim_counts) if victim_counts else 0
        final_believers = victim_counts[-1] if victim_counts else 0

        info_frame = ttk.Frame(content)
        info_frame.pack(fill='x', padx=8, pady=6)
        ttk.Label(info_frame, text=f'Topic: {getattr(self, "_sim_topic", "Unknown")}').pack(anchor='w')
        ttk.Label(info_frame, text=f'Juiciness: {self.juiciness.get()}/100').pack(anchor='w')

        # Small plots area
        plots_frame = ttk.Frame(content)
        plots_frame.pack(fill='both', expand=False, padx=4, pady=6)
        # Increase vertical size so x-axis title and ticks have room
        fig = plt.Figure(figsize=(10,4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if victim_counts:
            ax1.plot(range(1, len(victim_counts)+1), victim_counts, marker='o', color='#ff6b6b')
            # Use a clearer, bold title for ABM as requested and add axis labels
            ax1.set_title('Agent-Based Model (ABM)', fontweight='bold')
            ax1.set_xlabel('Round', fontsize=9)
            ax1.set_ylabel('Believers', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No ABM data', ha='center', va='center')

        try:
            pbm_hist = self.pbm_simulator.get_history()
            pbm_vals = pbm_hist.get('believers', [])
            if pbm_vals:
                ax2.plot(range(1, len(pbm_vals)+1), pbm_vals, marker='s', color='#6bafff')
                # PBM title and axis labels
                ax2.set_title('Population-Based Model (PBM)', fontweight='bold')
                ax2.set_xlabel('Round', fontsize=9)
                ax2.set_ylabel('Believers', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'No PBM data', ha='center', va='center')
        except Exception:
            ax2.text(0.5, 0.5, 'No PBM data', ha='center', va='center')

        canvas = FigureCanvasTkAgg(fig, master=plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Transmission log tree
        log_frame = ttk.Frame(content)
        log_frame.pack(fill='both', expand=True, padx=4, pady=6)
        ttk.Label(log_frame, text='Transmission Log (source → target)').pack(anchor='w')
        cols = ('round','source','target')
        tree = ttk.Treeview(log_frame, columns=cols, show='headings', height=8)
        for c in cols:
            tree.heading(c, text=c.title())
        tree.pack(fill='both', expand=True, pady=(4,4))
        for r, trans in enumerate(getattr(self.abm_simulator, 'transmission_history', []), start=1):
            for (s,t) in trans:
                tree.insert('', 'end', values=(r,s,t))

        # Spreaders per round (small text view)
        spread_frame = ttk.Frame(content)
        spread_frame.pack(fill='x', expand=False, padx=4, pady=6)
        ttk.Label(spread_frame, text='ABM Spreaders (per round)').pack(anchor='w')
        txt = tk.Text(spread_frame, height=6, wrap='none', font=('Courier', 9))
        txt.pack(fill='both', expand=True)
        spreaders_per_round = []
        for r, rnd in enumerate(self.round_history, start=1):
            ids = [i for i, v in enumerate(rnd) if v]
            spreaders_per_round.append(ids)
            txt.insert(tk.END, f'Round {r}: {ids}\n')
        txt.configure(state='disabled')
        
        # PBM visualization (right plot styling)
        if hasattr(self, 'pbm_simulator'):
            ax2.set_title("Population-Based Model (PBM)", pad=10, fontsize=10, fontweight='bold')
            ax2.set_xlabel("Round", fontsize=9)
            ax2.set_ylabel("Believers", fontsize=9)
            ax2.tick_params(axis='both', which='major', labelsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Update legend with improved styling
            ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            
            # Set y-axis limits to keep scale consistent
            ax2.set_ylim(0, len(self.abm_simulator.G.nodes()) * 1.1)
        
        # Adjust layout with specific padding
        self.fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)
        
        # Draw canvas
        self.canvas.draw()

    def _make_scrollable(self, parent):
        """Create a scrollable frame inside `parent` and return the inner frame.

        The returned frame is where callers should place their content. This
        helper clears existing children of `parent`.
        """
        # Clear existing content
        for w in parent.winfo_children():
            w.destroy()

        container = tk.Frame(parent)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = ttk.Frame(canvas)

        def _on_config(event):
            # This line configures the scroll region based on the inner frame's contents
            canvas.configure(scrollregion=canvas.bbox("all"))
            # NEW: If the width of the canvas changes, force the inner frame to match it
            canvas.itemconfig(canvas_window, width=event.width) # <--- ADD THIS LINE

        inner.bind('<Configure>', _on_config)
        # Store the ID of the created window so we can reference it
        canvas_window = canvas.create_window((0, 0), window=inner, anchor='nw') # <--- STORE WINDOW ID

        # NEW: Bind a handler to resize the inner frame whenever the CANVAS changes size (e.g., when the window is resized)
        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind('<Configure>', _on_canvas_resize) # <--- ADD THIS LINE
        # Enable scrolling with mouse wheel when hovering over inner frame
        def _on_mousewheel(event):
            # Windows / Mac differences
            delta = int(-1*(event.delta/120)) if event.delta else 0
            canvas.yview_scroll(delta, 'units')

        inner.bind_all('<MouseWheel>', _on_mousewheel)

        return inner
    def show_summary(self):
        """Populate the embedded Summary and Comparison tabs in the main window.

        This replaces the old popup windows and instead fills `self.tab_summary`
        and `self.tab_comparison`. It then selects the Summary tab for the user.
        """
        # Prepare results data structures for the visualizer
        history = self.scam_history if getattr(self, '_sim_topic', '') == "Financial Scam" else self.round_history
        abm_results = {
            'believer_counts': history,
            'total_agents': len(self.abm_simulator.G.nodes()) if hasattr(self, 'abm_simulator') else 0
        }
        try:
            pbm_results = self.pbm_simulator.get_history()
        except Exception:
            # fallback minimal structure
            pbm_results = {
                'susceptible': [0],
                'believers': [0],
                'immune': [0]
            }

        # Populate the Summary tab contents
        try:
            self._populate_summary_tab()
        except Exception:
            # best-effort: clear and show minimal message
            for w in self.tab_summary.winfo_children():
                w.destroy()
            ttk.Label(self.tab_summary, text='Summary not available', font=('Segoe UI', 11)).pack(padx=8, pady=8)

        # Populate the Comparison tab by embedding the ComparisonVisualizer
        for w in self.tab_comparison.winfo_children():
            w.destroy()
        try:
            from comparison_viz import ComparisonVisualizer
            viz = ComparisonVisualizer(abm_results, pbm_results, self.context_text.get(), intervention_rounds=self.intervention_rounds)
            comp_container = self._make_scrollable(self.tab_comparison)
            viz.show_comparison(comp_container)
        except Exception as e:
            # Show error message inside the tab instead of raising
            err_frame = ttk.Frame(self.tab_comparison)
            err_frame.pack(fill='both', expand=True, padx=10, pady=10)
            ttk.Label(err_frame, text='Failed to render comparison plots.', foreground='red').pack(anchor='w')
            ttk.Label(err_frame, text=str(e)).pack(anchor='w')

        # Switch to Summary tab for the user
        try:
            self.notebook.select(self.tab_summary)
        except Exception:
            pass

    def _show_scam_summary(self, summary_frame):
        """Show summary for scam simulation."""
        scam_counts = [sum(1 for a in round_data if a) 
                      for round_data in self.scam_history]
        total_scammed = sum(scam_counts)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(range(1, len(scam_counts)+1), scam_counts, marker='o', color='red', label='Victims')
        
        # Draw intervention lines
        for r in self.intervention_rounds:
            ax.axvline(
                r+1, color='green', linestyle='--',
                label='Intervention' if r == self.intervention_rounds[0] else None
            )
            
        # Always add legend if we have data
        if len(scam_counts) > 0:
            ax.legend(loc='upper left')
            
        ax.set_title("Scam Victims Over Time")
        ax.set_xlabel("Round")
        ax.set_ylabel("Victims")
        ax.grid(True, alpha=0.3)

        # Add plot to summary frame
        canvas = FigureCanvasTkAgg(fig, master=summary_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add summary text
        text_frame = tk.Frame(summary_frame)
        text_frame.pack(fill=tk.X, pady=5)
        summary = (
            f"Topic: {self._sim_topic}\n"
            f"Total Scam Victims: {total_scammed}\n"
            f"Peak Victims: {max(scam_counts) if scam_counts else 0}\n"
            f"Final Round State: {scam_counts[-1] if scam_counts else 0} victims"
        )

    def _show_fake_news_summary(self, summary_frame):
        """Show summary for fake news simulation."""
        # Calculate statistics
        victim_counts = [sum(1 for a in round_data if a) 
                        for round_data in self.round_history]
        total_shares = sum(victim_counts)
        peak_believers = max(victim_counts) if victim_counts else 0
        final_believers = victim_counts[-1] if victim_counts else 0
        peak_round = victim_counts.index(peak_believers) + 1 if victim_counts else 0
        total_agents = len(self.abm_simulator.G.nodes())
        
        # Calculate PBM statistics
        pbm_history = self.pbm_simulator.get_history()
        pbm_believers = pbm_history['believers']
        pbm_peak = max(pbm_believers)
        pbm_peak_round = pbm_believers.index(pbm_peak) + 1
        pbm_final = pbm_believers[-1]
        
        # Calculate intervention effects if applied
        pre_intervention = None
        post_intervention = None
        if self.intervention and len(victim_counts) > 5:
            pre_intervention = sum(victim_counts[:5]) / 5
            post_intervention = sum(victim_counts[5:10]) / 5
        
        # Calculate losses
        losses = self._calculate_losses(total_shares)
        
        # Create main info frame
        info_frame = tk.Frame(summary_frame, relief=tk.RIDGE, bd=2)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add context section
        tk.Label(info_frame, text="News Context Analysis", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        context_text = (
            f"Topic Category: {self._sim_topic_category}\n"
            f"Context: {self.context_text.get()}\n"
            f"Juiciness Score: {self.juiciness.get()}/100\n"
            f"Topic Weight: {self._sim_topic_weight:.2f}"
        )
        tk.Label(info_frame, text=context_text, justify=tk.LEFT, font=('Arial', 10), wraplength=500).pack(padx=10, pady=5)
        
        # Add model comparison section
        tk.Label(info_frame, text="Model Comparison", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        comparison_text = (
            f"Agent-Based Model (ABM):\n"
            f"• Peak Believers: {peak_believers} ({peak_believers/total_agents*100:.1f}%) at round {peak_round}\n"
            f"• Final State: {final_believers} ({final_believers/total_agents*100:.1f}%)\n"
            f"• Total Shares: {total_shares}\n\n"
            f"Population-Based Model (PBM):\n"
            f"• Peak Believers: {int(pbm_peak)} ({pbm_peak/total_agents*100:.1f}%) at round {pbm_peak_round}\n"
            f"• Final State: {int(pbm_final)} ({pbm_final/total_agents*100:.1f}%)\n"
            f"• Diffusion Rate: {pbm_peak/total_agents:.3f}"
        )
        tk.Label(info_frame, text=comparison_text, justify=tk.LEFT, font=('Arial', 10)).pack(padx=10, pady=5)
        
        # Add intervention effects if applicable
        if pre_intervention is not None:
            tk.Label(info_frame, text="Intervention Analysis", font=('Arial', 12, 'bold')).pack(pady=(10,5))
            intervention_text = (
                f"Pre-intervention average: {pre_intervention:.1f} believers/round\n"
                f"Post-intervention average: {post_intervention:.1f} believers/round\n"
                f"Effect: {((post_intervention - pre_intervention) / pre_intervention * 100):.1f}% change"
            )
            tk.Label(info_frame, text=intervention_text, justify=tk.LEFT, font=('Arial', 10)).pack(padx=10, pady=5)
        
        # Add risk analysis section
        tk.Label(info_frame, text="Risk Analysis", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        risk_text = (
            f"Financial Impact Risk: {losses['Financial Loss']*100:.1f}%\n"
            f"Reputational Damage Risk: {losses['Reputation Loss']*100:.1f}%\n"
            f"Trust Erosion Risk: {losses['Trust Loss']*100:.1f}%"
        )
        tk.Label(info_frame, text=risk_text, justify=tk.LEFT, font=('Arial', 10)).pack(padx=10, pady=5)
        
        # ===== Interactive ABM spreaders & transmission log =====
        trans_frame = tk.LabelFrame(summary_frame, text="ABM Spreaders & Transmission Log", padx=8, pady=8)
        trans_frame.pack(fill=tk.BOTH, padx=10, pady=8)

        # Left: Spreaders per round list + details
        left = tk.Frame(trans_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))

        tk.Label(left, text="ABM Spreaders (per round)", font=('Arial', 11, 'bold')).pack(anchor='w')
        round_listbox = tk.Listbox(left, height=8)
        round_listbox.pack(fill=tk.BOTH, expand=False, pady=(4,4))

        # populate rounds
        spreaders_by_round = []
        for r, rnd in enumerate(self.round_history, start=1):
            ids = [i for i, v in enumerate(rnd) if v]
            spreaders_by_round.append(ids)
            round_listbox.insert(tk.END, f"Round {r}: {len(ids)} spreader(s)")

        # detail box for selected round
        detail_label = tk.Label(left, text="Selected round spreaders:")
        detail_label.pack(anchor='w')
        detail_txt = tk.Text(left, height=4, wrap='none')
        detail_txt.pack(fill=tk.BOTH, expand=False)

        total_unique = set(x for ids in spreaders_by_round for x in ids)
        uniq_label = tk.Label(left, text=f"Total unique spreaders: {len(total_unique)}")
        uniq_label.pack(anchor='w', pady=(4,0))

        # Show a compact, scrollable list of unique spreader IDs
        ids_list = sorted(total_unique)
        display_ids = ids_list[:500]
        ids_frame = tk.Frame(left)
        ids_frame.pack(fill='x', pady=(4,6))
        tk.Label(ids_frame, text='Unique spreaders (IDs):', font=('Arial', 9)).pack(anchor='w')
        # Use a Listbox with vertical scrollbar so the unique spreaders are easy to scan
        lb_frame = tk.Frame(ids_frame)
        lb_frame.pack(fill='both', expand=True)
        ids_listbox = tk.Listbox(lb_frame, height=6, selectmode=tk.SINGLE, font=('Courier', 9))
        ids_scroll = ttk.Scrollbar(lb_frame, orient='vertical', command=ids_listbox.yview)
        ids_listbox.configure(yscrollcommand=ids_scroll.set)
        ids_listbox.pack(side='left', fill='both', expand=True)
        ids_scroll.pack(side='right', fill='y')
        if display_ids:
            for i in display_ids:
                ids_listbox.insert(tk.END, str(i))
            if len(ids_list) > len(display_ids):
                ids_listbox.insert(tk.END, f'...(+{len(ids_list)-len(display_ids)} more)')
        else:
            ids_listbox.insert(tk.END, 'No spreaders recorded.')

        # Double-clicking an ID in the unique spreaders list will highlight
        # that node in the ABM visualization for quick inspection.
        def _on_ids_double_click(event):
            sel = event.widget.curselection()
            if not sel:
                return
            val = event.widget.get(sel[0])
            # ignore the "...(+N more)" entry
            if isinstance(val, str) and val.startswith('...'):
                return
            try:
                node_id = int(val)
            except Exception:
                return
            try:
                # switch to Visualize tab and call update_graph with highlight
                self.notebook.select(self.tab_visualize)
            except Exception:
                pass
            try:
                self.update_graph(highlight_ids=[node_id])
                # revert highlight after 4 seconds to avoid sticky state
                self.root.after(4000, lambda: self.update_graph())
            except Exception:
                pass

        ids_listbox.bind('<Double-Button-1>', _on_ids_double_click)

        def on_round_select(evt):
            sel = round_listbox.curselection()
            detail_txt.delete('1.0', tk.END)
            if not sel:
                return
            idx = sel[0]
            ids = spreaders_by_round[idx]
            detail_txt.insert(tk.END, f"IDs: {ids}\n")
            detail_txt.insert(tk.END, f"Count: {len(ids)}\n")

        round_listbox.bind('<<ListboxSelect>>', on_round_select)

        # Right: Transmission log (treeview) + controls
        right = tk.Frame(trans_frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Transmission Log (source → target)", font=('Arial', 11, 'bold')).pack(anchor='w')
        tree = ttk.Treeview(right, columns=('round', 'source', 'target'), show='headings', height=8)
        tree.heading('round', text='Round')
        tree.heading('source', text='Source')
        tree.heading('target', text='Target')
        tree.column('round', width=60, anchor='center')
        tree.column('source', width=80, anchor='center')
        tree.column('target', width=80, anchor='center')
        tree.pack(fill=tk.BOTH, expand=True, pady=(4,4))

        # populate transmission entries
        total_trans = 0
        for r, trans in enumerate(getattr(self.abm_simulator, 'transmission_history', []), start=1):
            for (s, t) in trans:
                tree.insert('', tk.END, values=(r, s, t))
                total_trans += 1

        total_label = tk.Label(right, text=f"Total transmissions: {total_trans}")
        total_label.pack(anchor='w', pady=(4,2))

        def export_transmissions():
            # Ask for filename
            fname = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv')])
            if not fname:
                return
            try:
                with open(fname, 'w', newline='') as fh:
                    fh.write('round,source,target\n')
                    for r, trans in enumerate(getattr(self.abm_simulator, 'transmission_history', []), start=1):
                        for (s, t) in trans:
                            fh.write(f"{r},{s},{t}\n")
                messagebox.showinfo('Export', f'Wrote transmissions to {fname}')
            except Exception as e:
                messagebox.showerror('Export error', str(e))

        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=(6,0))
        tk.Button(btn_frame, text='Export CSV', command=export_transmissions).pack(side=tk.LEFT)

        # neat small notes
        notes = tk.Label(summary_frame, text='Tip: select a round to see its spreaders. Use Export to save full transmission log.', fg='#333333')
        notes.pack(fill=tk.X, padx=12, pady=(2,8))
        # Add recommendations
        tk.Label(info_frame, text="Recommendations", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        recommendations = self._generate_recommendations(losses, peak_believers/total_agents)
        tk.Label(info_frame, text=recommendations, justify=tk.LEFT, font=('Arial', 10), wraplength=500).pack(padx=10, pady=(5,10))

        # --- Spreader information (ABM) ---
        spreader_frame = tk.Frame(summary_frame, relief=tk.RIDGE, bd=1)
        spreader_frame.pack(fill=tk.BOTH, padx=10, pady=(8, 10), expand=False)
        tk.Label(spreader_frame, text="ABM Spreaders (per round)", font=('Arial', 11, 'bold')).pack(anchor='w', padx=6, pady=(6,2))

        # Compute spreaders per round from ABM round history (self.round_history)
        # Each element in round_history is a list-like of booleans indicating shared state per agent
        spreaders_per_round = []
        unique_spreaders = set()
        for ridx, rnd in enumerate(self.round_history):
            try:
                spreaders = [i for i, val in enumerate(rnd) if val]
            except Exception:
                # If rnd is not iterable in the expected way, skip
                spreaders = []
            spreaders_per_round.append(spreaders)
            unique_spreaders.update(spreaders)

        # Create a small scrollable text widget to list spreaders
        text_container = tk.Frame(spreader_frame)
        text_container.pack(fill=tk.BOTH, padx=6, pady=(0,6), expand=True)
        txt = tk.Text(text_container, height=6, wrap='none', font=('Courier', 9))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(text_container, orient='vertical', command=txt.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        txt.configure(yscrollcommand=scrollbar.set)

        if spreaders_per_round:
            for r, s in enumerate(spreaders_per_round, start=1):
                if s:
                    txt.insert(tk.END, f"Round {r}: {s}\n")
                else:
                    txt.insert(tk.END, f"Round {r}: []\n")
        else:
            txt.insert(tk.END, "No ABM rounds recorded (run the simulation first).\n")

        # Show cumulative unique spreaders summary
        txt.insert(tk.END, "\n")
        txt.insert(tk.END, f"Total unique spreaders: {len(unique_spreaders)}\n")
        if unique_spreaders:
            # show first 200 ids to avoid extremely long lists
            ids_list = sorted(unique_spreaders)
            display_ids = ids_list[:200]
            txt.insert(tk.END, f"Spreaders (IDs): {display_ids}")
            if len(ids_list) > len(display_ids):
                txt.insert(tk.END, f"  ...(+{len(ids_list)-len(display_ids)} more)\n")

        txt.configure(state='disabled')
        
        # --- Transmission log summary ---
        trans_frame = tk.Frame(summary_frame, relief=tk.RIDGE, bd=1)
        trans_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10), expand=False)
        tk.Label(trans_frame, text="Transmission Log (source -> target)", font=('Arial', 11, 'bold')).pack(anchor='w', padx=6, pady=(6,2))

        trans_text = tk.Text(trans_frame, height=6, wrap='none', font=('Courier', 9))
        trans_text.pack(fill=tk.BOTH, padx=6, pady=(0,6))
        # Get transmissions from simulator if available
        transmissions_all = []
        if hasattr(self, 'abm_simulator') and hasattr(self.abm_simulator, 'transmission_history'):
            transmissions_all = self.abm_simulator.transmission_history

        if transmissions_all:
            total_edges = sum(len(r) for r in transmissions_all)
            trans_text.insert(tk.END, f"Total transmissions: {total_edges}\n\n")
            # show per-round transmissions
            for r, edges in enumerate(transmissions_all, start=1):
                if edges:
                    trans_text.insert(tk.END, f"Round {r}: {edges}\n")
                else:
                    trans_text.insert(tk.END, f"Round {r}: []\n")
        else:
            trans_text.insert(tk.END, "No transmission records available. Run the simulation to generate logs.\n")

        trans_text.configure(state='disabled')

    def _generate_recommendations(self, losses, peak_spread_rate):
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Spread rate based recommendations
        if peak_spread_rate > 0.7:
            recommendations.append("URGENT: Immediate response required - viral spread detected")
        elif peak_spread_rate > 0.4:
            recommendations.append("High spread rate - prioritize containment measures")
        
        # Risk-based recommendations
        max_risk = max(losses.items(), key=lambda x: x[1])
        if max_risk[1] > 0.6:
            recommendations.append(f"Critical {max_risk[0].lower()} risk - implement mitigation strategies")
        
        # General recommendations
        if self.intervention:
            recommendations.append("Continue monitoring intervention effectiveness")
        else:
            recommendations.append("Consider implementing intervention measures")
            
        return "\n".join(f"• {r}" for r in recommendations)

    def _calculate_losses(self, total_shares):
        """Calculate various loss probabilities."""
        juice_factor = self.juiciness.get() / 100.0
        financial_loss = reputational_loss = trust_loss = 0.05

        if self._sim_topic_category == 'Reputation-based Rumors':
            reputational_loss = min(1.0, 0.1 + 0.025 * total_shares * juice_factor)
        elif self._sim_topic_category in ['Phishing', 'Scare Tactics']:
            trust_loss = min(1.0, 0.15 + 0.035 * total_shares * 
                           (1 if not self.intervention else 0.5))
        elif (self._sim_topic_category == 'Policy Manipulation' or 
              self._sim_topic in ["Financial Scam", "Fake Scholarship Scam",
                                "Emergency VPN Update", "Student Aid Sabotage",
                                "University Database Hacked", "Data Breach"]):
            financial_loss = min(1.0, 0.2 + 0.012 * total_shares + 
                               0.007 * self.juiciness.get())

        return {
            "Financial Loss": financial_loss,
            "Reputation Loss": reputational_loss,
            "Trust Loss": trust_loss
        }

    def _create_summary_plots(self, victim_counts, losses):
        """Create summary visualization plots."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        # Victims over time plot
        axs[0].plot(range(1, len(victim_counts)+1), victim_counts, marker='o')
        for r in self.intervention_rounds:
            axs[0].axvline(
                r+1, color='green', linestyle='--',
                label='Intervention' if r == self.intervention_rounds[0] else ""
            )
        if self.intervention_rounds:
            axs[0].legend()
        axs[0].set_title("Victims Over Time")
        axs[0].set_xlabel("Round")
        axs[0].set_ylabel("Victims")

        # Loss probabilities plot
        bar_colors = ["#E15759", "#F28E2B", "#4E79A7"]
        bars = axs[1].bar(losses.keys(), losses.values(), color=bar_colors)
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel("Probability (0-1)")
        axs[1].set_title("Simulated Loss Probabilities (Higher = Worse)")

        # Add value labels
        for bar in bars:
            xpos = bar.get_x() + bar.get_width() / 2
            value = bar.get_height()
            axs[1].text(xpos, value - 0.03, f"{value:.2f}", 
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', color='white')

        fig.tight_layout(rect=[0, 0.18, 1, 1])
        fig.subplots_adjust(bottom=0.28)
        fig.text(0.5, 0.13, 
                "Each bar shows the risk of each loss type from this simulation "
                "(0 = none, 1 = certain)",
                ha='center', fontsize=9, color='#333')

        return fig

    def _add_summary_controls(self, summary_win, fig):
        """Add control buttons to summary window."""
        def add_more_rounds():
            self.max_rounds += 10
            self.extra_rounds = True
            plt.close(fig)
            summary_win.destroy()
            self.root.after(500, self.automate_rounds)

        def end_simulation_now():
            plt.close(fig)
            summary_win.destroy()
            try:
                self.reset_btn.state(['!disabled'])
            except Exception:
                pass

        button_frame = tk.Frame(summary_win)
        button_frame.pack(pady=8)
        
        more_btn = tk.Button(button_frame, text="Add 10 More Rounds",
                            command=add_more_rounds)
        more_btn.pack(side="left", padx=5)
        
        end_btn = tk.Button(button_frame, text="End Simulation Now",
                           command=end_simulation_now)
        end_btn.pack(side="left", padx=5)

        def on_close():
            plt.close(fig)
            summary_win.destroy()
            try:
                self.reset_btn.state(['!disabled'])
            except Exception:
                pass
        summary_win.protocol("WM_DELETE_WINDOW", on_close)

    def _init_simulators(self):
        """Initialize both ABM and PBM simulators."""
        # Reset simulation state
        self.round = 0
        self.round_label.config(text=f"Round: {self.round}")
        self.intervention = False
        self.extra_rounds = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []

        # Initialize ABM simulator
        num_agents = len(self.df_agents)
        # ABM must use the CSV agent profiles
        self.abm_simulator = FakeNewsSimulator(num_agents, self.df_agents)
        self.abm_simulator.seed_initial_state(is_scam=(self._sim_topic == "Financial Scam"))

        # Apply ABM sharing model parameters from GUI controls (if present)
        try:
            # These variables are optional - only set if the GUI provided them
            self.abm_simulator.share_belief_weight = float(getattr(self, 'share_belief_var', tk.DoubleVar(4.0)).get())
            self.abm_simulator.share_emotional_weight = float(getattr(self, 'share_emotion_var', tk.DoubleVar(1.5)).get())
            self.abm_simulator.share_confirmation_weight = float(getattr(self, 'share_conf_var', tk.DoubleVar(1.0)).get())
            self.abm_simulator.share_juice_weight = float(getattr(self, 'share_juice_var', tk.DoubleVar(1.0)).get())
            self.abm_simulator.share_critical_penalty = float(getattr(self, 'share_crit_var', tk.DoubleVar(2.0)).get())
            self.abm_simulator.share_offset = float(getattr(self, 'share_offset_var', tk.DoubleVar(-1.0)).get())

            # Map the attribution combo human label to internal mode
            sel = getattr(self, 'attribution_var', tk.StringVar('Weighted by source belief')).get()
            mapping = {
                'Weighted by source belief': 'weighted',
                'Random among sharers': 'random',
                'First sharer': 'first'
            }
            self.abm_simulator.attribution_mode = mapping.get(sel, 'weighted')
        except Exception:
            # If any variable is missing or invalid, leave simulator defaults intact
            pass

        # Initialize PBM: use random-average agents by default (each agent = column means)
        means = self.df_agents.mean(numeric_only=True)
        # Repeat the mean row for num_agents rows to create average agents
        pbm_agent_df = pd.DataFrame([means.values] * num_agents, columns=means.index)

        # Estimate initial believer probability per PBM agent using a simple heuristic
        # (mirrors ABM trait influence in a compact form)
        cb = pbm_agent_df.get('confirmation_bias', pd.Series(0.5, index=pbm_agent_df.index)).astype(float)
        es = pbm_agent_df.get('emotional_susceptibility', pd.Series(0.5, index=pbm_agent_df.index)).astype(float)
        tl = pbm_agent_df.get('trust_level', pd.Series(0.5, index=pbm_agent_df.index)).astype(float)
        ct = pbm_agent_df.get('critical_thinking', pd.Series(0.5, index=pbm_agent_df.index)).astype(float)

        juice = self.juiciness.get() / 100.0
        topic_w = self._sim_topic_weight
        # Probability heuristic
        P = 0.3 + 0.2 * topic_w + 0.15 * ((cb + es + tl) / 3.0) - 0.2 * ct
        P = np.clip(P, 0.0, 1.0)
        initial_believers_pbm = int(round(P.sum()))

        # Allow user override for PBM initial believers
        try:
            user_init = int(self.pbm_initial_believers_var.get())
            if user_init > 0:
                initial_believers_pbm = user_init
            elif user_init == 0:
                initial_believers_pbm = 0
        except Exception:
            pass

        # PBM state trackers
        self.pbm_believers = [initial_believers_pbm / max(1, num_agents)]
        self.pbm_history = []

        # --- PBM simulator independent initialization ---
        # Pass the estimated initial believers into the PopulationSimulator
        self.pbm_simulator = PopulationSimulator(num_agents, initial_believers=initial_believers_pbm)

        # Adjust PBM rates based on simulation conditions (topic weight, juiciness, intervention)
        self.pbm_simulator.adjust_rates(
            self._sim_topic_weight,
            juice,
            self.intervention
        )

        # Override PBM rates with user-specified parameters (if provided)
        try:
            cr = float(self.pbm_contact_rate_var.get())
            br = float(self.pbm_belief_rate_var.get())
            rr = float(self.pbm_recovery_rate_var.get())
            # Apply if sensible
            if 0.0 <= cr <= 1.0:
                self.pbm_simulator.rates.contact_rate = cr
            if 0.0 <= br <= 1.0:
                # Some models store belief_rate as 'belief_rate' or params; try both
                setattr(self.pbm_simulator.rates, 'belief_rate', br)
            if 0.0 <= rr <= 1.0:
                setattr(self.pbm_simulator.rates, 'recovery_rate', rr)
        except Exception:
            pass

        # --- Calibration: scale PBM contact_rate to account for ABM network structure ---
        # Compute average degree (avg number of neighbors) in the ABM network
        try:
            degrees = [d for _, d in self.abm_simulator.G.degree()]
            avg_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
            # Network density = avg_deg / (N-1)
            density = avg_deg / max(1.0, (num_agents - 1))
            # Scale factor: modest amplification as density increases
            mult = 3.0
            try:
                mult = float(self.calibration_multiplier_var.get())
            except Exception:
                pass
            scale = 1.0 + mult * density
            # Apply scaling to PBM contact rate (keep within reasonable bound)
            original = getattr(self.pbm_simulator.rates, 'contact_rate', None)
            if original is not None:
                self.pbm_simulator.rates.contact_rate = min(0.95, original * scale)
        except Exception:
            # If anything goes wrong, leave PBM rates as-is
            pass

        # Initialize results containers with initial state (use estimated PBM initial believers)
        if self._sim_topic == "Financial Scam":
            initial_believers_abm = sum(1 for i in self.abm_simulator.agent_states 
                                if self.abm_simulator.agent_states[i]['scammed'])
        else:
            initial_believers_abm = sum(1 for i in self.abm_simulator.agent_states 
                                if self.abm_simulator.agent_states[i]['shared'])

        self.abm_results = {
            'believer_counts': [initial_believers_abm],
            'total_agents': num_agents
        }

        # Initialize PBM results with the initial counts we computed
        self.pbm_results = {
            'susceptible': [num_agents - initial_believers_pbm],
            'believers': [initial_believers_pbm],
            'immune': [0]
        }

        # Update UI: enable reset, disable run
        try:
            self.reset_btn.state(['!disabled'])
            self.run_btn.state(['disabled'])
        except Exception:
            pass
        self.update_graph()
        self.root.after(500, self.automate_rounds)

    def run_next_round(self):
        """Run the next simulation round for both models."""
        if self.round >= self.max_rounds:
            return

        # Run ABM simulation step
        if self._sim_topic == "Financial Scam":
            abm_result = self.abm_simulator.simulate_scam_round()
            self.scam_history.append(abm_result)
            self.abm_results['believer_counts'].append(
                sum(1 for x in abm_result if x)
            )
        else:
            abm_result = self.abm_simulator.simulate_fake_news_round(
                self.juiciness.get() / 100.0,
                self._sim_topic_weight,
                self._sim_topic_category,
                self.intervention,
                getattr(self, 'extra_rounds', False)
            )
            self.round_history.append(abm_result)
            # Store the actual count of believers
            believer_count = sum(1 for x in abm_result if x)
            self.abm_results['believer_counts'].append(believer_count)

        # Run PBM simulation step
        if self.intervention:
            self.pbm_simulator.adjust_rates(
                self._sim_topic_weight,
                self.juiciness.get() / 100.0,
                True
            )
        susceptible, believers, immune = self.pbm_simulator.simulate_step()
        self.pbm_results['susceptible'].append(susceptible)
        self.pbm_results['believers'].append(believers)
        self.pbm_results['immune'].append(immune)

        # Update round counter and UI
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}")
        
        if self.round >= self.max_rounds:
            self.show_summary()
            
        self.update_graph()

    def show_summary(self):
        """Populate the embedded Summary and Comparison tabs in the main window.

        This replaces the old popup windows and instead fills `self.tab_summary`
        and `self.tab_comparison`. It then selects the Summary tab for the user.
        """
        # Prepare results data structures for the visualizer
        history = self.scam_history if getattr(self, '_sim_topic', '') == "Financial Scam" else self.round_history
        abm_results = {
            'believer_counts': history,
            'total_agents': len(self.abm_simulator.G.nodes()) if hasattr(self, 'abm_simulator') else 0
        }
        try:
            pbm_results = self.pbm_simulator.get_history()
        except Exception:
            # fallback minimal structure
            pbm_results = {
                'susceptible': [0],
                'believers': [0],
                'immune': [0]
            }

        # Populate the Summary tab contents
        try:
            self._populate_summary_tab()
        except Exception:
            # best-effort: clear and show minimal message
            for w in self.tab_summary.winfo_children():
                w.destroy()
            ttk.Label(self.tab_summary, text='Summary not available', font=('Segoe UI', 11)).pack(padx=8, pady=8)

        # Populate the Comparison tab by embedding the ComparisonVisualizer
        for w in self.tab_comparison.winfo_children():
            w.destroy()
        try:
            from comparison_viz import ComparisonVisualizer
            viz = ComparisonVisualizer(abm_results, pbm_results, self.context_text.get(), intervention_rounds=self.intervention_rounds)
            comp_container = self._make_scrollable(self.tab_comparison)
            viz.show_comparison(comp_container)
        except Exception as e:
            # Show error message inside the tab instead of raising
            err_frame = ttk.Frame(self.tab_comparison)
            err_frame.pack(fill='both', expand=True, padx=10, pady=10)
            ttk.Label(err_frame, text='Failed to render comparison plots.', foreground='red').pack(anchor='w')
            ttk.Label(err_frame, text=str(e)).pack(anchor='w')

        # Switch to Summary tab for the user
        try:
            self.notebook.select(self.tab_summary)
        except Exception:
            pass

    def reset_simulation(self):
        """Reset the simulation state."""
        self.round = 0
        self.max_rounds = 10
        try:
            self.reset_btn.state(['disabled'])
            self.run_btn.state(['!disabled'])
        except Exception:
            pass
        self.ax.clear()
        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()
        self.round_label.config(text="Round: 0")
        self.intervention = False