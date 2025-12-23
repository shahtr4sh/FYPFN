"""Visualization utilities for comparing ABM and PBM results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ComparisonVisualizer:
    def __init__(self, abm_results: Dict, pbm_results: Dict, context: str, intervention_rounds=None):
        """Initialize the comparison visualizer.
        
        Args:
            abm_results: Results from agent-based simulation
            pbm_results: Results from population-based simulation
            context: The news context used in simulation
        """
        self.abm_results = abm_results
        self.pbm_results = pbm_results
        self.context = context
        # intervention_rounds: optional list of rounds where interventions occurred
        self.intervention_rounds = intervention_rounds or []
        
    def show_comparison(self, window) -> None:
        """Display comparison visualizations in the provided container.

        The `window` argument may be a Tk `Toplevel` or any Tkinter container (Frame).
        If it supports `geometry()` (i.e., Toplevel), the method will set window size.
        """
        # Create main container frame
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create figure with subplots; increase figure size so each subplot
        # receives more pixels and the donut can be rendered at a comfortable
        # scale without being tiny.
        fig = plt.figure(figsize=(14, 9))

        # Time series comparison
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        self._plot_time_series(ax1)

        # Final state comparison
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        self._plot_final_state(ax2)

        # Metrics comparison
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        self._plot_metrics_comparison(ax3)

        # Create canvas and pack it (do this before connecting hover events so
        # the event handler can call canvas.draw_idle()).
        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.06, wspace=0.25, hspace=0.35)
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Interactive hover annotation setup ---
        # Create per-axis annotation objects so annotations use the
        # correct coordinate systems and appear attached to the axis
        # being hovered (prevents the tooltip from showing on the
        # wrong subplot).
    # Put tooltip text to the left of the point so the arrow appears on
    # the left side of the label (use negative x-offset and right-align)
        hover_ann_ts = ax1.annotate("", xy=(0,0), xytext=(-30,15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"),
                    horizontalalignment='right')
        hover_ann_ts.set_visible(False)

        hover_ann_pie = ax2.annotate("", xy=(0,0), xytext=(-30,15), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"),
                     horizontalalignment='right')
        hover_ann_pie.set_visible(False)

        hover_ann_bar = ax3.annotate("", xy=(0,0), xytext=(-30,15), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"),
                     horizontalalignment='right')
        hover_ann_bar.set_visible(False)

        def _on_move(event):
            # clear default
            vis = False
            try:
                if event.inaxes == ax1:
                    # time series: check ABM scatter first
                    sc = getattr(self, '_abm_scatter', None)
                    if sc is not None:
                        hit, info = sc.contains(event)
                        if hit:
                            inds = info.get('ind', [])
                            if inds:
                                i = inds[0]
                                x = self._abm_rounds[i]
                                y = self._abm_counts[i]
                                hover_ann_ts.xy = (x, y)
                                hover_ann_ts.set_text(f'Round: {x}\nBelievers: {y}')
                                hover_ann_ts.set_visible(True)
                                # hide other anns
                                hover_ann_pie.set_visible(False)
                                hover_ann_bar.set_visible(False)
                                vis = True
                                canvas.draw_idle()
                                return
                    # PBM scatter
                    sc2 = getattr(self, '_pbm_scatter', None)
                    if sc2 is not None:
                        hit, info = sc2.contains(event)
                        if hit:
                            inds = info.get('ind', [])
                            if inds:
                                i = inds[0]
                                x = self._pbm_rounds[i]
                                y = self._pbm_believers[i]
                                hover_ann_ts.xy = (x, y)
                                hover_ann_ts.set_text(f'Round: {x}\nPBM believers: {y}')
                                hover_ann_ts.set_visible(True)
                                hover_ann_pie.set_visible(False)
                                hover_ann_bar.set_visible(False)
                                vis = True
                                canvas.draw_idle()
                                return

                elif event.inaxes == ax2:
                    # pie wedges: iterate wedges
                    wedges = getattr(self, '_pie_wedges', [])
                    labels = getattr(self, '_pie_labels', [])
                    for w, lab in zip(wedges, labels):
                        hit, _ = w.contains(event)
                        if hit:
                            # use pie-axis annotation; event.xdata/ydata are
                            # in data coords for that axis, so attach the
                            # annotation to ax2.
                            hover_ann_pie.xy = (event.xdata or 0, event.ydata or 0)
                            hover_ann_pie.set_text(str(lab))
                            hover_ann_pie.set_visible(True)
                            hover_ann_ts.set_visible(False)
                            hover_ann_bar.set_visible(False)
                            vis = True
                            canvas.draw_idle()
                            return

                elif event.inaxes == ax3:
                    # bars: check rectangles
                    for rect, metric_label, val in getattr(self, '_metric_bars', []):
                        hit, _ = rect.contains(event)
                        if hit:
                            hover_ann_bar.xy = (event.xdata or 0, event.ydata or 0)
                            hover_ann_bar.set_text(f'{metric_label}: {val}')
                            hover_ann_bar.set_visible(True)
                            hover_ann_ts.set_visible(False)
                            hover_ann_pie.set_visible(False)
                            vis = True
                            canvas.draw_idle()
                            return
            except Exception:
                pass

            if not vis:
                hover_ann_ts.set_visible(False)
                canvas.draw_idle()

        # connect hover handler
        canvas.mpl_connect('motion_notify_event', _on_move)

        # Make the matplotlib figure responsive to container resize so it
        # fills the available space. Adjust figure size in inches based on
        # the canvas widget pixel size and figure dpi.
        def _on_resize(event):
            try:
                w = event.width
                h = event.height
                if w <= 0 or h <= 0:
                    return
                dpi = fig.get_dpi()
                # compute size in inches, clamp to sensible minimums
                new_w = max(4.0, w / float(dpi))
                new_h = max(3.0, h / float(dpi))
                fig.set_size_inches(new_w, new_h, forward=True)
                canvas.draw_idle()
            except Exception:
                pass

        # Bind resize on the canvas widget's container
        canvas_widget.bind('<Configure>', _on_resize)

        # Add text summary below
        summary_frame = tk.Frame(main_frame)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        self._add_text_summary(summary_frame)

        # If window is a Toplevel, try to set a reasonable geometry; ignore otherwise
        try:
            if hasattr(window, 'geometry'):
                window.geometry("1200x800")
        except Exception:
            # ignore if the container doesn't support geometry
            pass
    def _plot_time_series(self, ax):
        """Plot time series comparison."""
        # Calculate ABM believer counts
        abm_history = self.abm_results['believer_counts']
        abm_counts = [sum(1 for x in round_data if x) for round_data in abm_history]
        rounds = range(len(abm_counts))
        
        # Get PBM believer counts
        pbm_believers = self.pbm_results['believers']
        
        # Plot results
        ax.plot(rounds, abm_counts, 'b-', label='ABM Believers', linewidth=2)
        ax.plot(rounds, pbm_believers[:len(rounds)], 'r--', label='PBM Believers', linewidth=2)
        
        # Add points for better visibility and keep references for hover
        abm_sc = ax.scatter(rounds, abm_counts, color='blue', s=30, alpha=0.6)
        pbm_sc = ax.scatter(rounds, pbm_believers[:len(rounds)], color='red', s=30, alpha=0.6)

        # Store artist references and underlying data for hover interaction
        self._abm_scatter = abm_sc
        self._abm_rounds = list(rounds)
        self._abm_counts = list(abm_counts)
        self._pbm_scatter = pbm_sc
        self._pbm_rounds = list(rounds)
        # ensure pbm_believers is a list with same length
        self._pbm_believers = list(pbm_believers[:len(rounds)])
        
        ax.set_title('Belief Spread Over Time: ABM vs PBM', pad=10)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Number of Believers', fontsize=10)
        
        # --- FIX: Only draw intervention line if self.intervention_rounds is not empty ---
        ivs = self.intervention_rounds
        
        first = True
        for r in ivs:
            # draw at integer round value
            ax.axvline(x=r, color='green', linestyle=':', linewidth=2, label='Intervention' if first else None)
            first = False

        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
    def _plot_final_state(self, ax):
        """Plot final state comparison pie charts."""
        abm_final_believers = sum(1 for x in self.abm_results['believer_counts'][-1] if x)
        abm_final = [
            abm_final_believers,
            self.abm_results['total_agents'] - abm_final_believers
        ]
        
        # Get all final PBM counts
        final_believers = int(self.pbm_results['believers'][-1])
        final_susceptible = int(self.pbm_results['susceptible'][-1])
        final_immune = int(self.pbm_results.get('immune', [0])[-1]) # Get 'immune', default to 0
        
        # Non-believers are both Susceptible AND Immune
        final_non_believers = final_susceptible + final_immune
        
        pbm_final = [
            final_believers,
            final_non_believers
        ]
        
        # Create mini pie charts (donut style) and show percentages on both rings.
        # Colors: red for believers, blue for non-believers (darker outer, lighter inner)
        outer_colors = ['red', 'blue']  # [believers, non-believers]
        inner_colors = ['lightcoral', 'lightblue']

        # helper to make compact autopct with a prefix per wedge
        def make_autopct(prefixes):
            idx = {'i': 0}
            def _autopct(pct):
                try:
                    label = prefixes[idx['i']]
                except Exception:
                    label = ''
                idx['i'] += 1
                # compact wording: prefix followed by percentage with one decimal
                return f"{label} {pct:.1f}%"
            return _autopct

        # Outer ring (ABM) — show ABM percentages per wedge. Reduce radii
        # slightly so the chart fits comfortably within its axes even when
        # the canvas is narrow.
        wedges1, texts1, autotexts1 = ax.pie(
            abm_final, radius=1.0, center=(0, 0),
            colors=outer_colors,
            labels=None,
            autopct=make_autopct("  "),
            pctdistance=0.75,
            wedgeprops=dict(width=0.22, edgecolor='white')
        )

        # Inner ring (PBM) — slightly smaller inner radius to preserve spacing
        wedges2, texts2, autotexts2 = ax.pie(
            pbm_final, radius=0.7, center=(0, 0),
            colors=inner_colors,
            labels=None,
            autopct=make_autopct(""),
            pctdistance=0.6,
            wedgeprops=dict(width=0.2, edgecolor='white')
        )

        # Tidy text appearance: smaller, bold for readability in the donut
        for t in autotexts1 + autotexts2:
            t.set_fontsize(8)
            t.set_fontweight('bold')
            t.set_color('black')

        ax.set_title('Final State Comparison\nOuter: ABM, Inner: PBM')

        # Keep the donut circular and expand axis limits based on the reduced
        # radius so the outer ring and labels are never clipped.
        ax.set_aspect('equal')
        pad = 1.0
        try:
            ax.set_xlim(-pad, pad)
            ax.set_ylim(-pad, pad)
        except Exception:
            pass

        # Add legend mapping colors to labels (use the darker shades for legend clarity)
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color='blue', label='Non-believers'),
            mpatches.Patch(color='red', label='Believers')
        ]
        # Place legend outside to the upper-right; adjusting bbox_to_anchor so
        # it sits outside the reduced right subplot boundary. For narrow
        # canvases we center the legend below the pie so it doesn't overlap
        # the donut.
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=9, borderaxespad=0., ncol=2)
        # expose wedges and labels for hover interaction
        try:
            # --- FIX: Store Inner (PBM) first, then Outer (ABM) ---
            # We check PBM first so the Outer Ring doesn't "block" the hover event
            self._pie_wedges = list(wedges2) + list(wedges1)
            
            # Match the labels to the new order (PBM first, then ABM)
            self._pie_labels = [
                f'PBM believers: {pbm_final[0]}', f'PBM non-believers: {pbm_final[1]}',
                f'ABM believers: {abm_final[0]}', f'ABM non-believers: {abm_final[1]}'
            ]
        except Exception:
            self._pie_wedges = []
            self._pie_labels = []
        
    def _plot_metrics_comparison(self, ax):
        """Plot comparison of key metrics."""
        metrics = ['Peak Believers', 'Spread Rate', 'Time to Peak']
        
        # Calculate ABM metrics
        abm_counts = [sum(1 for x in round_data if x) 
                     for round_data in self.abm_results['believer_counts']]
        abm_max_believers = max(abm_counts)
        abm_peak_time = np.argmax(abm_counts)
        total_agents = self.abm_results['total_agents']
        
        abm_values = [
            abm_max_believers,
            abm_max_believers / total_agents,
            abm_peak_time
        ]
        
        # Calculate PBM metrics
        pbm_believers = np.array(self.pbm_results['believers'])
        pbm_max = max(pbm_believers)
        
        pbm_values = [
            pbm_max,
            pbm_max / total_agents,  # Use same total agents as ABM
            np.argmax(pbm_believers)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        rects_abm = ax.bar(x - width/2, abm_values, width, label='ABM', color='blue', alpha=0.6)
        rects_pbm = ax.bar(x + width/2, pbm_values, width, label='PBM', color='red', alpha=0.6)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_title('Comparison of Key Metrics')

        # expose bar rectangles for hover; store as list of (rect, label, value)
        try:
            self._metric_bars = []
            # rects_abm/rects_pbm are BarContainer objects; iterate their rectangles
            for i, rect in enumerate(rects_abm):
                self._metric_bars.append((rect, metrics[i] + ' (ABM)', abm_values[i]))
            for i, rect in enumerate(rects_pbm):
                self._metric_bars.append((rect, metrics[i] + ' (PBM)', pbm_values[i]))
        except Exception:
            self._metric_bars = []
        
    def _add_text_summary(self, frame):
        """Add text summary of comparison."""
        # Calculate statistics
        abm_peak = max(sum(1 for x in round_data if x) 
                      for round_data in self.abm_results['believer_counts'])
        pbm_peak = max(self.pbm_results['believers'])
        abm_final = sum(1 for x in self.abm_results['believer_counts'][-1] if x)
        pbm_final = int(self.pbm_results['believers'][-1])
        total_agents = self.abm_results['total_agents']
        
        # Calculate time to peak and convergence
        abm_peak_time = next(i for i, round_data in enumerate(self.abm_results['believer_counts']) 
                            if sum(1 for x in round_data if x) == abm_peak)
        pbm_peak_time = next(i for i, believers in enumerate(self.pbm_results['believers']) 
                            if believers == pbm_peak)
        
        # Create header frame with title
        header_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2, bg='#f0f0f0')
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        # Main title
        tk.Label(header_frame, text="Simulation Analysis", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(pady=5)
        
        # Context section
        context_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2)
        context_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(context_frame, text="Context Analysis", font=('Arial', 11, 'bold')).pack(pady=5)
        tk.Label(context_frame, text=self.context, font=('Arial', 10), wraplength=400).pack(pady=5)
        
        # Statistics frame
        stats_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(stats_frame, text="Model Performance Metrics", font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Create two columns
        cols_frame = tk.Frame(stats_frame)
        cols_frame.pack(fill=tk.X, padx=10, pady=5)
        left_col = tk.Frame(cols_frame)
        right_col = tk.Frame(cols_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ABM Statistics (Left Column)
        tk.Label(left_col, text="Agent-Based Model (ABM)", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        tk.Label(left_col, text=f"• Peak Believers: {abm_peak} ({(abm_peak/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(left_col, text=f"• Time to Peak: Round {abm_peak_time}", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(left_col, text=f"• Final State: {abm_final} believers ({(abm_final/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        
        # PBM Statistics (Right Column)
        tk.Label(right_col, text="Population-Based Model (PBM)", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        tk.Label(right_col, text=f"• Peak Believers: {pbm_peak} ({(pbm_peak/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(right_col, text=f"• Time to Peak: Round {pbm_peak_time}", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(right_col, text=f"• Final State: {pbm_final} believers ({(pbm_final/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        
        # Analysis frame
        analysis_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(analysis_frame, text="Comparative Analysis", font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Model comparison analysis
        comparison_text = f"""• Diffusion Speed: {'ABM' if abm_peak_time < pbm_peak_time else 'PBM'} shows faster initial spread
• Peak Difference: {abs(abm_peak - pbm_peak)} agents ({abs(abm_peak/total_agents*100 - pbm_peak/total_agents*100):.1f}%)
• Final State Difference: {abs(abm_final - pbm_final)} agents ({abs(abm_final/total_agents*100 - pbm_final/total_agents*100):.1f}%)
• Model Characteristics:
  - ABM: Captures individual variations and network effects
  - PBM: Shows smoother, population-level dynamics"""
        
        tk.Label(analysis_frame, text=comparison_text, font=('Arial', 10), justify=tk.LEFT).pack(padx=10, pady=5, anchor='w')
