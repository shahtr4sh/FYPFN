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
        self.juiciness = tk.IntVar(value=50) # This is already 0-100
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
        self.simulation_params_snapshot = {} # Holds the params of the last run
        
        # Load agents data
        # ABM CSV chooser (default will be chosen in setup_ui)
        self.abm_csv_var = tk.StringVar()
        self._load_agent_data()

        # --- NEW PERCENTAGE-ONLY SLIDER VARIABLES ---
        # These store the slider's position (e.g., -100 to 200)
        self.pbm_contact_rate_pct_var = tk.DoubleVar(value=0.0)
        self.pbm_belief_rate_pct_var = tk.DoubleVar(value=0.0)
        self.pbm_recovery_rate_pct_var = tk.DoubleVar(value=0.0)
        
        self.share_belief_pct_var = tk.DoubleVar(value=0.0)
        self.share_emotion_pct_var = tk.DoubleVar(value=0.0)
        self.share_conf_pct_var = tk.DoubleVar(value=0.0)
        self.share_juice_pct_var = tk.DoubleVar(value=0.0)
        
        self.calibration_multiplier_pct_var = tk.DoubleVar(value=0.0)

        # We also store the default values here for easy access
        self.DEFAULTS = {
            'pbm_contact': 0.4,
            'pbm_belief': 0.3,
            'pbm_recovery': 0.1,
            'share_belief': 4.0,
            'share_emotion': 1.5,
            'share_conf': 1.0,
            'share_juice': 1.0,
            'calibration_multiplier': 3.0
        }
        
        # This one is for the juiciness label
        self.juiciness_label_var = tk.StringVar(value=f"Value: {self.juiciness.get()}%")

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
                    # This will trigger the trace to update the label
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
            # Helper to reverse-calculate the percentage for a slider
            def _set_pct_var(pct_var, default_val, actual_val):
                try:
                    actual = float(actual_val)
                    if default_val == 0:
                        pct_var.set(0.0)
                    else:
                        percent = ((actual / default_val) - 1.0) * 100.0
                        pct_var.set(percent)
                except Exception:
                    pass # Keep slider at default if value is invalid

            # --- NEW: Set percentage sliders from scenario values ---
            for k in ['share_belief_weight', 'share_belief_var']:
                if k in abm_params:
                    _set_pct_var(self.share_belief_pct_var, self.DEFAULTS['share_belief'], abm_params[k])
                    break
            
            for k in ['share_emotional_weight', 'share_emotion_var']:
                if k in abm_params:
                    _set_pct_var(self.share_emotion_pct_var, self.DEFAULTS['share_emotion'], abm_params[k])
                    break
            
            for k in ['share_conf_weight', 'share_confirmation_weight', 'share_conf_var']:
                if k in abm_params:
                    _set_pct_var(self.share_conf_pct_var, self.DEFAULTS['share_conf'], abm_params[k])
                    break
            
            for k in ['share_juice_weight', 'share_juice_var']:
                if k in abm_params:
                    _set_pct_var(self.share_juice_pct_var, self.DEFAULTS['share_juice'], abm_params[k])
                    break

            # Note: crit_var and offset_var removed from your new GUI, so no need to set them.

            if 'attribution_mode' in abm_params:
                # set human-label combo directly if present
                try:
                    self.attribution_var.set(abm_params.get('attribution_mode'))
                except Exception:
                    pass

            # PBM rates
            pbm = data.get('pbm_rates', {}) or {}
            try:
                # Set the PERCENT vars; the 'command' will update the label
                if 'contact_rate' in pbm:
                     _set_pct_var(self.pbm_contact_rate_pct_var, self.DEFAULTS['pbm_contact'], pbm['contact_rate'])
                if 'belief_rate' in pbm:
                    _set_pct_var(self.pbm_belief_rate_pct_var, self.DEFAULTS['pbm_belief'], pbm['belief_rate'])
                if 'recovery_rate' in pbm:
                    _set_pct_var(self.pbm_recovery_rate_pct_var, self.DEFAULTS['pbm_recovery'], pbm['recovery_rate'])
                if 'initial_believers' in pbm:
                    self.pbm_initial_believers_var.set(int(pbm.get('initial_believers')))
            except Exception:
                pass

            # advanced params
            adv = data.get('advanced_params', {}) or {}
            try:
                if 'calibration_multiplier' in adv:
                    _set_pct_var(self.calibration_multiplier_pct_var, self.DEFAULTS['calibration_multiplier'], adv['calibration_multiplier'])
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
            
            # Use getattr() to safely get _sim_topic, providing 'scenario' as a default
            suggested_topic = getattr(self, '_sim_topic', 'scenario')
            suggested = (suggested_topic or 'scenario').replace(' ', '_') + '.json'
            
            fname = filedialog.asksaveasfilename(initialdir=self.scenarios_dir, initialfile=suggested, defaultextension='.json', filetypes=[('JSON files','*.json')])
            if not fname:
                return

            # --- FIX: Helper to read the REAL values from the sliders ---
            def _calc_val(default, pct_var):
                pct = pct_var.get()
                val = default + (default * (pct / 100.0))
                return max(0.0, val)

            scenario = {
                'name': os.path.splitext(os.path.basename(fname))[0],
                'description': '',
                'context': self.context_text.get(),
                'abm_csv': self.abm_csv_var.get(),
                'juiciness': int(self.juiciness.get()),
                'max_rounds': int(self.max_rounds_var.get()),
                'intervention_round': int(self.intervention_round_var.get()),
                'abm_sharing_params': {
                    'share_belief_weight': _calc_val(self.DEFAULTS['share_belief'], self.share_belief_pct_var),
                    'share_emotional_weight': _calc_val(self.DEFAULTS['share_emotion'], self.share_emotion_pct_var),
                    'share_conf_weight': _calc_val(self.DEFAULTS['share_conf'], self.share_conf_pct_var),
                    'share_juice_weight': _calc_val(self.DEFAULTS['share_juice'], self.share_juice_pct_var),
                    'attribution_mode': self.attribution_var.get()
                },
                
                'pbm_rates': {
                    'contact_rate': _calc_val(self.DEFAULTS['pbm_contact'], self.pbm_contact_rate_pct_var),
                    'belief_rate': _calc_val(self.DEFAULTS['pbm_belief'], self.pbm_belief_rate_pct_var),
                    'recovery_rate': _calc_val(self.DEFAULTS['pbm_recovery'], self.pbm_recovery_rate_pct_var),
                    'initial_believers': int(self.pbm_initial_believers_var.get())
                },
                'advanced_params': {
                    'calibration_multiplier': _calc_val(self.DEFAULTS['calibration_multiplier'], self.calibration_multiplier_pct_var)
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

    def _create_relative_slider(self, parent_frame, text, percent_var, 
                                default_value, min_percent=-100, max_percent=200, 
                                format_str=".2f"):
        """
        Creates a slider that adjusts a percentage variable.
        Uses a trace to ensure the label updates even when setting values via code (scenarios).
        """
        ttk.Label(parent_frame, text=text, font=('Segoe UI', 10)).pack(anchor='w', padx=8)
        
        # This label will show the calculated value
        label_var = tk.StringVar()
        
        # Internal update function
        def _update_label(*args):
            try:
                # Get value (works if arg is string from command or float from var)
                try:
                    percent = percent_var.get()
                except:
                    return

                actual = default_value + (default_value * (percent / 100.0))
                actual = max(0.0, actual)
                
                # Sanitize format string just in case curly braces were left in
                fmt = format_str.replace("{", "").replace("}", "")
                
                # Update the label text
                label_var.set(f"Value: {actual:{fmt}}  ({percent:+.0f}%)")
            except Exception:
                label_var.set("Value: ---")

        # Create the slider
        # We remove 'command' because the trace below handles EVERYTHING now.
        slider = tk.Scale(parent_frame, from_=min_percent, to=max_percent, orient=tk.HORIZONTAL,
                          variable=percent_var, length=260, showvalue=1, resolution=1)
        slider.pack(anchor='w', padx=8)

        # The label that displays the value
        label = ttk.Label(parent_frame, textvariable=label_var, font=('Segoe UI', 9))
        label.pack(anchor='w', padx=10, pady=(0, 8))

        # --- THE FIX: Bind the variable changes to the label update ---
        # This ensures that if you load a scenario (variable.set), the label updates too.
        percent_var.trace_add('write', _update_label)

        # Initialize text immediately
        _update_label()
        
    def browse_agent_file(self):
        """Allows user to select any CSV file from their computer."""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Agent Profile CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if filename:
            # 1. Store the full path for the simulation to use
            self.abm_csv_var.set(filename)

            # 2. Update the dropdown visual text to show the file name
            short_name = os.path.basename(filename)
            self.abm_csv_combo.set(short_name)
            
            # 3. Configure the combobox values to include this new file
            current_values = list(self.abm_csv_combo.cget("values"))
            if short_name not in current_values:
                current_values.append(short_name)
                self.abm_csv_combo.configure(values=current_values)
                
            # Reload data immediately
            self._load_agent_data()
            print(f"Loaded Agent File: {filename}")
        
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
        
        # --- FIX: Create a container frame to hold dropdown + button side-by-side ---
        csv_container = ttk.Frame(params_frame)
        csv_container.pack(anchor='w', padx=8, pady=(0,6), fill='x')

        # populate CSV list
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        try:
            csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
        except FileNotFoundError:
            csv_files = []
            
        self.abm_csv_var.set(self.abm_csv_var.get() or (csv_files[0] if csv_files else ''))
        
        # Create Dropdown inside the container (Left side)
        self.abm_csv_combo = ttk.Combobox(csv_container, values=csv_files, textvariable=self.abm_csv_var, state='readonly', width=25)
        self.abm_csv_combo.pack(side='left', padx=(0, 5))
        self.abm_csv_combo.bind('<<ComboboxSelected>>', lambda e: self._load_agent_data())

        # Create Browse Button inside the container (Right side)
        self.btn_browse_csv = ttk.Button(
            csv_container,
            text="Browse...", 
            width=8, 
            command=self.browse_agent_file
        )
        self.btn_browse_csv.pack(side='left')

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

        # Juiciness slider (already 0-100, but we add the label)
        ttk.Label(params_frame, text='Juiciness (%):', font=default_font).pack(anchor='w', padx=8)
        # --- MODIFICATION: Set showvalue=1 ---
        self.juiciness_scale = tk.Scale(params_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                        variable=self.juiciness, length=260, showvalue=1)
        self.juiciness_scale.pack(anchor='w', padx=8)
        
        # Add the label for Juiciness
        def _update_juiciness_label(*args):
            self.juiciness_label_var.set(f"Value: {self.juiciness.get()}%")
        self.juiciness.trace_add('write', _update_juiciness_label)
        ttk.Label(params_frame, textvariable=self.juiciness_label_var, font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=(0, 8))
        _update_juiciness_label() # Set initial value

        ttk.Separator(params_frame, orient='horizontal').pack(fill='x', pady=6, padx=6)

        # Group: PBM params
        ttk.Label(params_frame, text='Population Model (PBM)', font=heading_font).pack(anchor='w', padx=8, pady=(6,4))

        # Create the new 0-100% sliders
        self._create_relative_slider(params_frame, "PBM contact rate (%):", 
            self.pbm_contact_rate_pct_var, default_value=self.DEFAULTS['pbm_contact'], 
            min_percent=-100, max_percent=200, format_str=".2f")
            
        self._create_relative_slider(params_frame, "PBM belief rate (%):", 
            self.pbm_belief_rate_pct_var, default_value=self.DEFAULTS['pbm_belief'], 
            min_percent=-100, max_percent=200, format_str=".2f")
            
        self._create_relative_slider(params_frame, "PBM recovery rate (%):", 
            self.pbm_recovery_rate_pct_var, default_value=self.DEFAULTS['pbm_recovery'], 
            min_percent=-100, max_percent=200, format_str=".2f")
        
        ttk.Label(params_frame, text='PBM initial believers:', font=default_font).pack(anchor='w', padx=8)
        self.pbm_initial_believers_var = tk.IntVar(value=5)
        tk.Spinbox(params_frame, from_=0, to=10000, textvariable=self.pbm_initial_believers_var, width=12).pack(anchor='w', padx=8, pady=(0,8))

        ttk.Separator(params_frame, orient='horizontal').pack(fill='x', pady=6, padx=6)

        # Group: ABM sharing model controls
        ttk.Label(params_frame, text='Agent-Based Model (ABM)', font=heading_font).pack(anchor='w', padx=8, pady=(6,4))
        ttk.Label(params_frame, text='*Higher weights indicate more influence.', font=note_font).pack(anchor='w', padx=8, pady=(1,1))
        
        self._create_relative_slider(params_frame, "Belief Weight (%):", 
            self.share_belief_pct_var, default_value=self.DEFAULTS['share_belief'], 
            min_percent=-100, max_percent=100, format_str=".1f")
            
        self._create_relative_slider(params_frame, "Emotional Weight (%):", 
            self.share_emotion_pct_var, default_value=self.DEFAULTS['share_emotion'], 
            min_percent=-100, max_percent=200, format_str=".1f")
            
        self._create_relative_slider(params_frame, "Confirmation Bias Weight (%):", 
            self.share_conf_pct_var, default_value=self.DEFAULTS['share_conf'], 
            min_percent=-100, max_percent=200, format_str=".1f")
            
        self._create_relative_slider(params_frame, "Virality Multiplier (%):", 
            self.share_juice_pct_var, default_value=self.DEFAULTS['share_juice'], 
            min_percent=-100, max_percent=200, format_str=".2f")
        
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
        
        self._create_relative_slider(params_frame, "Calibration multiplier (%):", 
            self.calibration_multiplier_pct_var, default_value=self.DEFAULTS['calibration_multiplier'], 
            min_percent=-100, max_percent=200, format_str=".1f")
        
        # --- NEW: Intervention Toggle & Spinbox ---
        self.enable_intervention_var = tk.BooleanVar(value=True) # Default is ON
        self.intervention_round_var = tk.IntVar(value=5)

        # Container for the checkbutton and spinbox
        int_frame = ttk.Frame(params_frame)
        int_frame.pack(fill='x', padx=8, pady=(0, 8))

        # Function to enable/disable spinbox based on checkbox
        def _toggle_intervention():
            state = 'normal' if self.enable_intervention_var.get() else 'disabled'
            self.int_spin.configure(state=state)

        # The Checkbox
        chk = ttk.Checkbutton(int_frame, text='Intervention at Round:', 
                              variable=self.enable_intervention_var, 
                              command=_toggle_intervention)
        chk.pack(side='left')
        
        # The Spinbox (saved as self.int_spin so we can disable it)
        self.int_spin = tk.Spinbox(int_frame, from_=0, to=1000, 
                                   textvariable=self.intervention_round_var, width=6)
        self.int_spin.pack(side='left', padx=(5,0))
        ttk.Label(params_frame, text='Max rounds:', font=default_font).pack(anchor='w', padx=8)
        self.max_rounds_var = tk.IntVar(value=self.max_rounds)
        tk.Spinbox(params_frame, from_=1, to=1000, textvariable=self.max_rounds_var, width=12).pack(anchor='w', padx=8, pady=(0,12))

        # Run / Reset buttons (compact)
        btns = ttk.Frame(params_frame)
        btns.pack(fill='x', padx=8, pady=(6,12))
        self.run_btn = ttk.Button(btns, text='Run Simulation', command=self.init_simulation)
        self.run_btn.pack(side='left', expand=True, fill='x', padx=(0,6))
        self.rerun_btn = ttk.Button(btns, text="Run Again", command=self.rerun_simulation, state="disabled")
        self.rerun_btn.pack(side=tk.LEFT, padx=5)
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
        
        # --- NEW: Calculate final values from percentages ---
        def _calc_val(default, pct_var):
            pct = pct_var.get()
            val = default + (default * (pct / 100.0))
            return max(0.0, val) # Ensure no negative values

        try:
            # Read the percentages and calculate the final values
            val_belief = _calc_val(self.DEFAULTS['share_belief'], self.share_belief_pct_var)
            val_emotion = _calc_val(self.DEFAULTS['share_emotion'], self.share_emotion_pct_var)
            val_conf = _calc_val(self.DEFAULTS['share_conf'], self.share_conf_pct_var)
            val_juice = _calc_val(self.DEFAULTS['share_juice'], self.share_juice_pct_var)
            
            val_pbm_contact = _calc_val(self.DEFAULTS['pbm_contact'], self.pbm_contact_rate_pct_var)
            val_pbm_belief = _calc_val(self.DEFAULTS['pbm_belief'], self.pbm_belief_rate_pct_var)
            val_pbm_recovery = _calc_val(self.DEFAULTS['pbm_recovery'], self.pbm_recovery_rate_pct_var)
            
            val_calib = _calc_val(self.DEFAULTS['calibration_multiplier'], self.calibration_multiplier_pct_var)

            # Store a snapshot of the parameters used for this simulation
            self.simulation_params_snapshot = {
                'abm_belief_weight': val_belief,
                'abm_emotion_weight': val_emotion,
                'abm_conf_weight': val_conf,
                'abm_juice_weight': val_juice,
                'abm_attribution': self.attribution_var.get(),
                
                'pbm_contact': val_pbm_contact,
                'pbm_belief': val_pbm_belief,
                'pbm_recovery': val_pbm_recovery,
                'pbm_initial': int(self.pbm_initial_believers_var.get()),
                
                # Use a conditional expression to save -1 if disabled
                'run_intervention': int(self.intervention_round_var.get()) if self.enable_intervention_var.get() else -1,
                'run_calibration': val_calib
            }
            
        except Exception as e:
            print(f"Warning: Could not create params snapshot. {e}")
            self.simulation_params_snapshot = {} # Clear on error
            
        # Analyze context and set juiciness
        context = self.context_text.get().lower()
        juiciness_score = analyze_context_juiciness(context)
        self.juiciness.set(juiciness_score)
        # self.juiciness_scale.set(juiciness_score) # No longer needed, trace will do this

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
        if self.round > self.max_rounds:   # <--- Run UNTIL round is greater than max
            return
        
        # --- FIX: Check if the checkbox is checked! ---
        # Only run this logic if enable_intervention_var is True
        if self.enable_intervention_var.get():
            int_round = int(self.intervention_round_var.get())
            if self.round == int_round and not self.intervention:
                self.intervention = True
                self.intervention_rounds.append(self.round)
            
        # Run the next round
        self.run_next_round()
        
        # Schedule the next round after a delay
        if self.round < self.max_rounds:
            self.root.after(500, self.automate_rounds)

    def run_next_round(self):
        """Run the next simulation round for both models."""
        if self.round > self.max_rounds:   # <--- Run UNTIL round is greater than max
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
        
        # --- OPTIMIZATION: Use pre-calculated layout ---
        self.pos = getattr(self, 'fixed_layout', nx.spring_layout(self.abm_simulator.G, seed=42))
            
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
        
        content = self._make_scrollable(self.tab_summary)
        
        # Clear any existing content and use the Summary tab directly (no scrollable wrapper)
        for w in content.winfo_children(): # <-- Make sure this now uses 'content'
            w.destroy()
        # content = self.tab_summary # <-- DELETE OR COMMENT OUT THIS LINE

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

        # --- START: NEW PARAMETER SUMMARY SECTION ---
        
        params_summary_frame = ttk.LabelFrame(content, text="Simulation Parameters Used", padding=(10, 5))
        params_summary_frame.pack(fill='x', padx=8, pady=6)
        
        col_frame = ttk.Frame(params_summary_frame)
        col_frame.pack(fill='x')
        
        left_col = ttk.Frame(col_frame)
        left_col.pack(side='left', fill='x', expand=True, padx=5)
        
        right_col = ttk.Frame(col_frame)
        right_col.pack(side='left', fill='x', expand=True, padx=5)

        # --- Read from the SNAPSHOT, not the live variables ---
        params = self.simulation_params_snapshot
        
        # ABM Parameters (Left Column)
        ttk.Label(left_col, text="ABM Sharing Parameters:", font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        ttk.Label(left_col, text=f"  • Belief Weight: {params.get('abm_belief_weight', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(left_col, text=f"  • Emotional Weight: {params.get('abm_emotion_weight', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(left_col, text=f"  • Confirmation Bias: {params.get('abm_conf_weight', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(left_col, text=f"  • Virality Multiplier: {params.get('abm_juice_weight', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(left_col, text=f"  • Attribution: {params.get('abm_attribution', 'N/A')}").pack(anchor='w')
        
        # PBM & Advanced Parameters (Right Column)
        ttk.Label(right_col, text="PBM Parameters:", font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        ttk.Label(right_col, text=f"  • Contact Rate: {params.get('pbm_contact', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(right_col, text=f"  • Belief Rate: {params.get('pbm_belief', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(right_col, text=f"  • Recovery Rate: {params.get('pbm_recovery', 'N/A'):.2f}").pack(anchor='w')
        ttk.Label(right_col, text=f"  • Initial Believers: {params.get('pbm_initial', 'N/A')}").pack(anchor='w')
        
        ttk.Label(right_col, text="Run Settings:", font=('Segoe UI', 9, 'bold'), padding=(0, 5, 0, 0)).pack(anchor='w')
        ttk.Label(right_col, text=f"  • Intervention Round: {params.get('run_intervention', 'N/A')}").pack(anchor='w')
        ttk.Label(right_col, text=f"  • Calibration Multiplier: {params.get('run_calibration', 'N/A'):.1f}").pack(anchor='w')
        
        # --- END: NEW PARAMETER SUMMARY SECTION ---

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
                ax2.plot(range(1, len(pbm_vals)+1), pbm_vals, marker='s', color='#6bafff', label='Believers')
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
            
        # --- Enable Buttons at the end ---
        self.run_btn.state(['!disabled'])   # Enable Run
        self.reset_btn.state(['!disabled']) # Enable Reset
        self.rerun_btn.state(['!disabled']) # <--- NEW: Enable Run Again

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

    def _init_simulators(self):
        
        # Initialize ABM simulator
        num_agents = len(self.df_agents)
        # ABM must use the CSV agent profiles
        self.abm_simulator = FakeNewsSimulator(num_agents, self.df_agents)
        self.abm_simulator.seed_initial_state(is_scam=(self._sim_topic == "Financial Scam"))
        # --- OPTIMIZATION: Calculate layout once ---
        self.fixed_layout = nx.spring_layout(
            self.abm_simulator.G,
            k=4.0,          # Distance between nodes
            iterations=500, # Physics simulation steps
            scale=3.0,      # Size of the graph
            seed=42         # Fixed seed for consistency
        )

        # Apply ABM sharing model parameters from GUI controls (if present)
        try:
            # --- NEW: Calculate values from percentages ---
            def _calc_val(default, pct_var):
                pct = pct_var.get()
                val = default + (default * (pct / 100.0))
                return max(0.0, val)

            self.abm_simulator.share_belief_weight = _calc_val(self.DEFAULTS['share_belief'], self.share_belief_pct_var)
            self.abm_simulator.share_emotional_weight = _calc_val(self.DEFAULTS['share_emotion'], self.share_emotion_pct_var)
            self.abm_simulator.share_confirmation_weight = _calc_val(self.DEFAULTS['share_conf'], self.share_conf_pct_var)
            self.abm_simulator.share_juice_weight = _calc_val(self.DEFAULTS['share_juice'], self.share_juice_pct_var)
            
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
        initial_believers_pbm = int(round(P.sum())) # <-- This calculates 25

        # --- THIS IS THE FIX: RESTORED THE OVERRIDE BLOCK ---
        # Allow user override for PBM initial believers
        try:
            user_init = int(self.pbm_initial_believers_var.get())
            if user_init > 0:
                initial_believers_pbm = user_init  # <-- This overrides 25 with 5
            elif user_init == 0:
                initial_believers_pbm = 0
        except Exception:
            pass
        # --- END OF FIX ---

        # PBM state trackers
        self.pbm_believers = [initial_believers_pbm / max(1, num_agents)]
        self.pbm_history = []

        # --- PBM simulator independent initialization ---
        # Pass the estimated initial believers into the PopulationSimulator
        self.pbm_simulator = PopulationSimulator(num_agents, initial_believers=initial_believers_pbm) # <-- This now correctly gets 5

        # Adjust PBM rates based on simulation conditions (topic weight, juiciness, intervention)
        self.pbm_simulator.adjust_rates(
            self._sim_topic_weight,
            juice,
            self.intervention
        )

        # --- THIS IS THE CORRECTED SLIDER LOGIC, IN THE CORRECT PLACE ---
        # Override PBM rates with user-specified parameters (if provided)
        try:
            # --- NEW: Calculate values from percentages ---
            def _calc_val(default, pct_var):
                pct = pct_var.get()
                val = default + (default * (pct / 100.0))
                return max(0.0, val)

            self.pbm_simulator.rates.contact_rate = _calc_val(self.DEFAULTS['pbm_contact'], self.pbm_contact_rate_pct_var)
            setattr(self.pbm_simulator.rates, 'belief_rate', _calc_val(self.DEFAULTS['pbm_belief'], self.pbm_belief_rate_pct_var))
            setattr(self.pbm_simulator.rates, 'recovery_rate', _calc_val(self.DEFAULTS['pbm_recovery'], self.pbm_recovery_rate_pct_var))
            
        except Exception:
            pass
        # --- END OF FIX ---

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
                # --- THIS IS THE CORRECTED CALIBRATION LOGIC ---
                pct = self.calibration_multiplier_pct_var.get()
                default = self.DEFAULTS['calibration_multiplier']
                mult = default + (default * (pct / 100.0))
                mult = max(0.0, mult)
            except Exception:
                pass
            scale = 1.0 + mult * density
            # Apply scaling to PBM contact rate (keep within reasonable bound)
            original = getattr(self.pbm_simulator.rates, 'contact_rate', None)
            if original is not None:
                self.pbm_simulator.rates.contact_rate = original * scale
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
            self.run_btn.state(['!disabled'])
        except Exception:
            pass
        self.update_graph()
        self.root.after(500, self.automate_rounds)

    def reset_simulation(self):
        """Reset simulation state, sliders, text inputs, and file selections to default."""
        
        # 1. Reset Text Inputs
        self.context_text.set("")  # Clear the Fake News Context
        self.scenario_var.set("")  # Clear the selected Scenario name
        self.scenario_combo.set("") # Clear the Scenario dropdown display
        
        # 2. Reset Agent Profile to Default (Smart Logic)
        # We look for the file containing 'agent_profiles' again, just like startup
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")
            csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
            
            default_file = None
            for f in csv_files:
                if 'agent_profiles' in f.lower():
                    default_file = f
                    break
            
            # If found, set it and load it
            if default_file:
                self.abm_csv_var.set(default_file)
                self.abm_csv_combo.set(default_file)
                self._load_agent_data() # Important: Actually reload the dataframe!
        except Exception:
            pass # Use whatever is currently loaded if this fails

        # 3. Reset Sliders to 0.0
        # PBM Sliders
        self.pbm_contact_rate_pct_var.set(0.0)
        self.pbm_belief_rate_pct_var.set(0.0)
        self.pbm_recovery_rate_pct_var.set(0.0)
        
        # ABM Sliders
        self.share_belief_pct_var.set(0.0)
        self.share_emotion_pct_var.set(0.0)
        self.share_conf_pct_var.set(0.0)
        self.share_juice_pct_var.set(0.0)
        
        # Advanced Slider
        self.calibration_multiplier_pct_var.set(0.0)
        
        # 4. Reset Simulation Logic State
        self.round = 0
        try:
            self.reset_btn.state(['disabled'])
            self.run_btn.state(['!disabled'])
        except Exception:
            pass

        # 5. Clear Visualization & Data
        self.ax.clear()
        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()
        
        self.round_label.config(text="Round: 0")
        self.intervention = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []
        
        print("System fully reset to defaults.")
        
    def rerun_simulation(self):
        """
        Resets ONLY the simulation state (graph, rounds, history) 
        but KEEPS the current parameters (sliders, text, scenario).
        Then immediately triggers a new run.
        """
        print("Re-running simulation with current parameters...")

        # 1. Reset Simulation State (Counters & Data)
        self.round = 0
        self.intervention = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []
        
        # Reset Result Dictionaries
        self.abm_results = {'believer_counts': [], 'total_agents': 0}
        self.pbm_results = {'susceptible': [], 'believers': [], 'immune': []}

        # 2. Clear the Visualization
        self.ax.clear()
        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()
        self.round_label.config(text="Round: 0")
        
        # 3. Disable buttons briefly (standard practice)
        self.run_btn.state(['disabled'])
        self.rerun_btn.state(['disabled'])
        self.reset_btn.state(['disabled'])

        # 4. Trigger the standard Run function
        # This will pick up the current slider values/text automatically.
        self.init_simulation()