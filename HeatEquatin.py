# --- Imports ---
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import math
import time 
import numba 
import random 

# --- Tooltip Klasse ---
class Tooltip:
    """Klasse für Popup-Tooltips bei Hover über Widgets."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip, "+")
        self.widget.bind("<Leave>", self.hide_tooltip, "+")
        self.widget.bind("<FocusOut>", self.hide_tooltip, "+")
        self.widget.bind("<ButtonPress>", self.hide_tooltip, "+")

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()

# --- FTCS Schritt-Berechnungsfunktionen ---
@numba.njit(fastmath=True, parallel=True) 
def ftcs_step_numba(u_current, S_interior, alpha_x, alpha_y, dt, inv_dx2, inv_dy2, Ny_int, Nx_int):
    u_new_interior = np.empty((Ny_int, Nx_int), dtype=u_current.dtype)
    for i_rel in numba.prange(Ny_int): 
        for j_rel in range(Nx_int): 
            i_abs = i_rel + 1
            j_abs = j_rel + 1
            term_x = alpha_x * (u_current[i_abs, j_abs + 1] - 2 * u_current[i_abs, j_abs] + u_current[i_abs, j_abs - 1]) * inv_dx2
            term_y = alpha_y * (u_current[i_abs + 1, j_abs] - 2 * u_current[i_abs, j_abs] + u_current[i_abs - 1, j_abs]) * inv_dy2
            u_new_interior[i_rel, j_rel] = u_current[i_abs, j_abs] + dt * (term_x + term_y + S_interior[i_rel, j_rel])
    return u_new_interior

def ftcs_step_numpy(u_current, S_interior, alpha_x, alpha_y, dt, dx, dy, Ny_int, Nx_int):
    i_int_u, j_int_u = slice(1, Ny_int+1), slice(1, Nx_int+1)
    u_im1 = u_current[i_int_u, :-2]   
    u_ip1 = u_current[i_int_u, 2:]    
    u_jm1 = u_current[:-2,  j_int_u]  
    u_jp1 = u_current[2:,   j_int_u]  
    u_ij  = u_current[i_int_u, j_int_u]
    diffusion_x = 0.0
    if dx > 1e-12: diffusion_x = alpha_x * (u_ip1 - 2*u_ij + u_im1) / dx**2
    diffusion_y = 0.0
    if dy > 1e-12: diffusion_y = alpha_y * (u_jp1 - 2*u_ij + u_jm1) / dy**2
    u_new_interior = u_ij + dt * (diffusion_x + diffusion_y + S_interior)
    return u_new_interior

# --- GUI Klasse ---
class AdvancedHeatEquationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Erweiterte 2D Wärmeleitungssimulation v2.3") 
        master.geometry("1250x900") 

        self.grid_x = None; self.grid_y = None; self.temp_field = None
        self.anim = None; self.analysis_data = {'t': [], 'max_T': [], 'total_H': []}
        self.scatter_source = None; self.cbar = None
        self.T0 = 100.0

        self.solver_var = tk.StringVar(value="Explicit FTCS")
        self.bc_var = tk.StringVar(value="Neumann (Insulated)")
        self.vis_var = tk.StringVar(value="2D Heatmap (imshow)")
        
        # Variablen für Quelle 1
        self.source1_enabled_var = tk.BooleanVar(value=False)
        # Variablen für Quelle 2 
        self.source2_enabled_var = tk.BooleanVar(value=False)

        self.laplacian_method_var = tk.StringVar(value="Numba")

        control_frame = ttk.Frame(master, padding="10"); control_frame.pack(side=tk.TOP, fill=tk.X)
        plot_area_frame = ttk.Frame(master, padding="10"); plot_area_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.create_controls(control_frame)

        plot_frame_left = ttk.LabelFrame(plot_area_frame, text="Temperaturverteilung T(x, y, t)", padding="5")
        plot_frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame_left)
        self.canvas_widget = self.canvas.get_tk_widget(); self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        plot_frame_right = ttk.LabelFrame(plot_area_frame, text="Analyse", padding="5")
        plot_frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.fig_analysis, self.axs_analysis = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, master=plot_frame_right)
        self.canvas_analysis_widget = self.canvas_analysis.get_tk_widget(); self.canvas_analysis_widget.pack(fill=tk.BOTH, expand=True)
        self.line_max_T, = self.axs_analysis[0].plot([], [], 'r-', label='Max T(t)')
        self.line_total_H, = self.axs_analysis[1].plot([], [], 'b-', label='Gesamt H(t)')
        self._setup_analysis_plot()

        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.status_label.config(text="Bereit. Parameter prüfen & Simulation starten.")
        self._on_bc_change()
        self._toggle_source1_widgets() 
        self._toggle_source2_widgets()
        self._on_solver_change()
        self._on_vis_change()

    def _on_closing(self):
        print("Schließe Anwendung..."); self.stop_animation()
        try: plt.close(self.fig); plt.close(self.fig_analysis)
        except Exception as e: print(f"Fehler beim Schließen der Figuren: {e}")
        finally: self.master.destroy()

    def _setup_analysis_plot(self):
        self.axs_analysis[0].set_ylabel("Max Temp (°C)"); self.axs_analysis[0].grid(True); self.axs_analysis[0].legend(loc='best', fontsize='small')
        self.axs_analysis[1].set_xlabel("Zeit (s)"); self.axs_analysis[1].set_ylabel("Gesamtwärme (a.u.)"); self.axs_analysis[1].grid(True); self.axs_analysis[1].legend(loc='best', fontsize='small')
        self.fig_analysis.tight_layout()

    def create_controls(self, parent_frame):
        p1_frame = ttk.LabelFrame(parent_frame, text="Physik & Gitter"); p1_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        p2_frame = ttk.LabelFrame(parent_frame, text="Zeit & Anfang/Rand"); p2_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        p3_frame = ttk.LabelFrame(parent_frame, text="Quellen & Solver"); p3_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y) 
        p4_frame = ttk.LabelFrame(parent_frame, text="Visualisierung & Aktionen"); p4_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        status_frame = ttk.Frame(parent_frame); status_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        self.entries = {}
        self.source1_widgets = {} 
        self.source2_widgets = {} 

        tooltips = {
            "α_x": "Wärmeleitfähigkeit X (m²/s).\nPositiv. Groß -> schnell.", "α_y": "Wärmeleitfähigkeit Y (m²/s).\nPositiv. Groß -> schnell.",
            "Nx": "Gitterpunkte X (≥3).\nGroß -> genauer, langsamer.", "Ny": "Gitterpunkte Y (≥3).\nGroß -> genauer, langsamer.",
            "T_Spitze": "Anfangs-Temp. Spitze (°C).\nReelle Zahl.", "T_Rand": "Rand-Temp. (°C).\nNur bei Dirichlet aktiv.",
            "Zeit": "Simulationsdauer (s).\nPositiv.", "dt": "Zeitschritt (s).\nPositiv. Klein für Stabilität/Genauigkeit.",
            "Quelle1X": "Quelle 1: Position X (0-1).", "Quelle1Y": "Quelle 1: Position Y (0-1).",
            "Quelle1R": "Quelle 1: Radius (0-1).\n<=0 -> keine Quelle.", "Quelle1S": "Quelle 1: Stärke (°C/s).\nNegativ -> Senke.", 
            "Quelle2X": "Quelle 2: Position X (0-1).", "Quelle2Y": "Quelle 2: Position Y (0-1).", 
            "Quelle2R": "Quelle 2: Radius (0-1).\n<=0 -> keine Quelle.", "Quelle2S": "Quelle 2: Stärke (°C/s).\nNegativ -> Senke.", 
            "Solver": "Verfahren: Explizit(schnell, instabil), Implizit(langsam, stabil).",
            "Randbed.": "Randbedingung: Neumann(isoliert), Dirichlet(fest).",
            "Ansicht": "Darstellung: 2D Heatmap(schnell), 3D Oberfläche(langsamer).",
            "Quelle1 aktiv": "Aktiviert/Deaktiviert die Quelle 1.", 
            "Quelle2 aktiv": "Aktiviert/Deaktiviert die Quelle 2.",
            "LaplaceMethod": "Berechnungsmethode für expliziten FTCS-Solver:\nNumba: Sehr schnell (JIT-kompiliert)\nNumPy: Standard (vektorisiert)",
            "RandomParams": "Setzt alle Parameter auf zufällige, plausible Werte." 
        }

        # --- p1: Physik & Gitter ---
        labels1 = ["α_x:", "α_y:", "Nx:", "Ny:"]; defaults1 = ["0.01", "0.01", "50", "50"]
        for i, label_text in enumerate(labels1):
            key = label_text.replace(":", ""); lbl = ttk.Label(p1_frame, text=label_text); lbl.grid(row=i, column=0, padx=2, pady=2, sticky="w")
            Tooltip(lbl, tooltips[key]); entry = ttk.Entry(p1_frame, width=7); entry.grid(row=i, column=1, padx=2, pady=2, sticky="ew")
            entry.insert(0, defaults1[i]); self.entries[key] = entry; Tooltip(entry, tooltips[key])

        # --- p2: Zeit & Anfang/Rand ---
        labels2 = ["T_Spitze (°C):", "T_Rand (°C):", "Zeit (s):", "dt (s):"]; defaults2 = ["100.0", "0.0", "1.0", "0.001"]
        for i, label_text in enumerate(labels2):
            lbl = ttk.Label(p2_frame, text=label_text); lbl.grid(row=i, column=0, padx=2, pady=2, sticky="w")
            key = label_text.split()[0].replace("(", ""); Tooltip(lbl, tooltips[key]); entry = ttk.Entry(p2_frame, width=7)
            entry.grid(row=i, column=1, padx=2, pady=2, sticky="ew"); entry.insert(0, defaults2[i]); self.entries[key] = entry
            Tooltip(entry, tooltips[key])
        lbl_bc = ttk.Label(p2_frame, text="Randbed.:"); lbl_bc.grid(row=len(labels2), column=0, padx=2, pady=2, sticky="w"); Tooltip(lbl_bc, tooltips["Randbed."])
        bc_combo = ttk.Combobox(p2_frame, textvariable=self.bc_var, values=["Neumann (Insulated)", "Dirichlet (Fixed Temp)"], width=18, state='readonly')
        bc_combo.grid(row=len(labels2), column=1, padx=2, pady=2, sticky="ew"); bc_combo.bind("<<ComboboxSelected>>", self._on_bc_change); Tooltip(bc_combo, tooltips["Randbed."])

        # --- p3: Quellen & Solver ---
        # Quelle 1
        q1_frame = ttk.LabelFrame(p3_frame, text="Quelle 1", padding=3); q1_frame.pack(fill=tk.X, padx=2, pady=2)
        source1_check = ttk.Checkbutton(q1_frame, text="Aktiv", variable=self.source1_enabled_var, command=self._toggle_source1_widgets) 
        source1_check.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="w"); Tooltip(source1_check, tooltips["Quelle1 aktiv"])
        labels_q1 = ["X:", "Y:", "R:", "S:"]; defaults_q1 = ["0.5", "0.5", "0.1", "10.0"]; keys_q1 = ["Quelle1X", "Quelle1Y", "Quelle1R", "Quelle1S"]
        for i, label_text in enumerate(labels_q1):
            lbl = ttk.Label(q1_frame, text=label_text, state=tk.DISABLED); lbl.grid(row=i+1, column=0, padx=2, pady=2, sticky="w")
            key = keys_q1[i]; Tooltip(lbl, tooltips[key]); entry = ttk.Entry(q1_frame, width=7, state=tk.DISABLED)
            entry.grid(row=i+1, column=1, padx=2, pady=2, sticky="ew"); entry.insert(0, defaults_q1[i]); self.entries[key] = entry
            self.source1_widgets[key] = (lbl, entry); Tooltip(entry, tooltips[key])

        # Quelle 2 
        q2_frame = ttk.LabelFrame(p3_frame, text="Quelle 2", padding=3); q2_frame.pack(fill=tk.X, padx=2, pady=(5,2))
        source2_check = ttk.Checkbutton(q2_frame, text="Aktiv", variable=self.source2_enabled_var, command=self._toggle_source2_widgets)
        source2_check.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="w"); Tooltip(source2_check, tooltips["Quelle2 aktiv"])
        labels_q2 = ["X:", "Y:", "R:", "S:"]; defaults_q2 = ["0.25", "0.75", "0.05", "-20.0"]; keys_q2 = ["Quelle2X", "Quelle2Y", "Quelle2R", "Quelle2S"]
        for i, label_text in enumerate(labels_q2):
            lbl = ttk.Label(q2_frame, text=label_text, state=tk.DISABLED); lbl.grid(row=i+1, column=0, padx=2, pady=2, sticky="w")
            key = keys_q2[i]; Tooltip(lbl, tooltips[key]); entry = ttk.Entry(q2_frame, width=7, state=tk.DISABLED)
            entry.grid(row=i+1, column=1, padx=2, pady=2, sticky="ew"); entry.insert(0, defaults_q2[i]); self.entries[key] = entry
            self.source2_widgets[key] = (lbl, entry); Tooltip(entry, tooltips[key])

        # Solver & FTCS Methode
        solver_frame = ttk.Frame(p3_frame); solver_frame.pack(fill=tk.X, padx=2, pady=(10,2))
        lbl_solver = ttk.Label(solver_frame, text="Solver:"); lbl_solver.grid(row=0, column=0, padx=2, pady=2, sticky="w"); Tooltip(lbl_solver, tooltips["Solver"])
        solver_combo = ttk.Combobox(solver_frame, textvariable=self.solver_var, values=["Explicit FTCS", "Implicit CN"], width=15, state='readonly')
        solver_combo.grid(row=0, column=1, padx=2, pady=2, sticky="ew"); Tooltip(solver_combo, tooltips["Solver"])
        solver_combo.bind("<<ComboboxSelected>>", self._on_solver_change)

        laplace_frame = ttk.Frame(solver_frame)
        laplace_frame.grid(row=1, column=0, columnspan=2, pady=(5,0), sticky="w")
        lbl_laplace = ttk.Label(laplace_frame, text="FTCS Schritt:")
        lbl_laplace.pack(side=tk.LEFT, padx=(0,5)); Tooltip(lbl_laplace, tooltips["LaplaceMethod"])
        self.rb_numba = ttk.Radiobutton(laplace_frame, text="Numba", variable=self.laplacian_method_var, value="Numba")
        self.rb_numba.pack(side=tk.LEFT); Tooltip(self.rb_numba, tooltips["LaplaceMethod"])
        self.rb_numpy = ttk.Radiobutton(laplace_frame, text="NumPy", variable=self.laplacian_method_var, value="NumPy")
        self.rb_numpy.pack(side=tk.LEFT, padx=(5,0)); Tooltip(self.rb_numpy, tooltips["LaplaceMethod"])

        # --- p4: Visualisierung & Aktionen ---
        lbl_vis = ttk.Label(p4_frame, text="Ansicht:"); lbl_vis.pack(anchor=tk.W, padx=2, pady=(5,0)); Tooltip(lbl_vis, tooltips["Ansicht"])
        vis_combo = ttk.Combobox(p4_frame, textvariable=self.vis_var, values=["2D Heatmap (imshow)", "3D Surface"], width=20, state='readonly')
        vis_combo.pack(anchor=tk.W, padx=2, pady=2); vis_combo.bind("<<ComboboxSelected>>", self._on_vis_change); Tooltip(vis_combo, tooltips["Ansicht"])
        self.start_button = ttk.Button(p4_frame, text="Start / Neu", command=self.start_simulation); self.start_button.pack(pady=5, fill=tk.X) #fill
        self.stop_button = ttk.Button(p4_frame, text="Stop", command=self.stop_animation, state=tk.DISABLED); self.stop_button.pack(pady=2, fill=tk.X) #fill
        self.reset_button = ttk.Button(p4_frame, text="Reset Params", command=self.reset_parameters); self.reset_button.pack(pady=2, fill=tk.X) #fill
        self.random_button = ttk.Button(p4_frame, text="Zufalls-Params", command=self.set_random_parameters); self.random_button.pack(pady=(5,2), fill=tk.X) 
        Tooltip(self.random_button, tooltips["RandomParams"])


        self.status_label = ttk.Label(status_frame, text="Bereit.", relief=tk.SUNKEN, anchor=tk.W); self.status_label.pack(side=tk.BOTTOM, fill=tk.X, expand=True, pady=(0,5))

    def _update_entry(self, key, value_str):
        """Hilfsfunktion zum Aktualisieren eines Entry-Widgets."""
        if key in self.entries:
            entry = self.entries[key]
            is_disabled = entry.cget('state') == tk.DISABLED
            if is_disabled: entry.config(state=tk.NORMAL)
            entry.delete(0, tk.END)
            entry.insert(0, value_str)
            if is_disabled: entry.config(state=is_disabled)
        else:
            print(f"Warnung: Entry-Schlüssel '{key}' nicht gefunden beim Update.")


    def set_random_parameters(self):
        """Setzt alle relevanten Parameter auf zufällige, plausible Werte."""
        self.stop_animation()
        
        # Solver
        solver = random.choice(["Explicit FTCS", "Implicit CN"])
        self.solver_var.set(solver)

        # Physik & Gitter
        alpha_x = random.uniform(0.001, 0.05)
        alpha_y = random.uniform(0.001, 0.05)
        Nx = random.randint(30, 70)
        Ny = random.randint(30, 70)
        self._update_entry("α_x", f"{alpha_x:.4f}")
        self._update_entry("α_y", f"{alpha_y:.4f}")
        self._update_entry("Nx", str(Nx))
        self._update_entry("Ny", str(Ny))

        # Zeit
        # dt Berechnung abhängig vom Solver
        if solver == "Explicit FTCS":
            # Stabilitätsbedingung beachten
            stable_dt_max = self._calculate_stable_dt(alpha_x, alpha_y, Nx, Ny)
            # Wähle dt sicher unter dem Maximum
            dt_val = random.uniform(stable_dt_max * 0.2, stable_dt_max * 0.7)
            dt_val = max(dt_val, 0.0001) # Mindest-dt
            # Gesamtzeit so, dass ca. 500-3000 Schritte entstehen
            num_steps = random.randint(500, 3000)
            zeit_val = num_steps * dt_val

        else: # Implicit CN
            dt_val = random.uniform(0.01, 0.5) # Größere dt möglich
            num_steps = random.randint(100, 500)
            zeit_val = num_steps * dt_val
        
        self._update_entry("Zeit", f"{zeit_val:.2f}")
        self._update_entry("dt", f"{dt_val:.4f}")

        # Anfang/Rand
        self.bc_var.set(random.choice(["Neumann (Insulated)", "Dirichlet (Fixed Temp)"]))
        self._update_entry("T_Spitze", f"{random.uniform(20.0, 150.0):.1f}")
        self._update_entry("T_Rand", f"{random.uniform(-10.0, 30.0):.1f}")

        # Quellen
        # Quelle 1
        self.source1_enabled_var.set(random.choice([True, False]))
        self._update_entry("Quelle1X", f"{random.uniform(0.1, 0.9):.2f}")
        self._update_entry("Quelle1Y", f"{random.uniform(0.1, 0.9):.2f}")
        self._update_entry("Quelle1R", f"{random.uniform(0.05, 0.2):.2f}")
        self._update_entry("Quelle1S", f"{random.uniform(-30.0, 30.0):.1f}")
        # Quelle 2
        self.source2_enabled_var.set(random.choice([True, False]))
        # Sicherstellen, dass Quelle 2 nicht identisch mit Quelle 1 ist 
        self._update_entry("Quelle2X", f"{random.uniform(0.1, 0.9):.2f}")
        self._update_entry("Quelle2Y", f"{random.uniform(0.1, 0.9):.2f}")
        self._update_entry("Quelle2R", f"{random.uniform(0.05, 0.15):.2f}")
        self._update_entry("Quelle2S", f"{random.uniform(-30.0, 30.0):.1f}")
        
        # FTCS Methode
        self.laplacian_method_var.set(random.choice(["Numba", "NumPy"]))
        # Visualisierung
        self.vis_var.set(random.choice(["2D Heatmap (imshow)", "3D Surface"]))

        # GUI Updates
        self.T0 = float(self.entries["T_Spitze"].get()) # self.T0 aktualisieren
        self._on_bc_change()
        self._toggle_source1_widgets()
        self._toggle_source2_widgets()
        self._on_solver_change()
        self._on_vis_change()
        
        # Plots leeren (wie in reset_parameters)
        try:
            if self.ax: self.ax.clear()
            self.axs_analysis[0].clear(); self.axs_analysis[1].clear()
            self._setup_analysis_plot()
            self.canvas.draw(); self.canvas_analysis.draw()
        except Exception as e: print(f"Fehler beim Leeren der Plots während Zufallsparams: {e}")

        self.status_label.config(text="Zufällige Parameter gesetzt. Bitte prüfen.")

    def _calculate_stable_dt(self, alpha_x_val, alpha_y_val, Nx_val, Ny_val):
        """Hilfsfunktion zur Berechnung des maximal stabilen dt für FTCS."""
        if Nx_val <= 1 or Ny_val <= 1: return 0.001 # Fallback
        dx = 1.0 / (Nx_val - 1)
        dy = 1.0 / (Ny_val - 1)
        if abs(dx) < 1e-9 or abs(dy) < 1e-9 : return 0.001 

        denominator_dx2 = dx**2
        denominator_dy2 = dy**2
        if abs(denominator_dx2) < 1e-12 or abs(denominator_dy2) < 1e-12 : return 0.001
        
        term_x = 0.0
        if alpha_x_val > 1e-9 : term_x = alpha_x_val / denominator_dx2
        
        term_y = 0.0
        if alpha_y_val > 1e-9 : term_y = alpha_y_val / denominator_dy2

        if term_x + term_y < 1e-9: # Wenn keine Diffusion oder riesiges Gitter
            return 10.0 # Kann ein sehr großes dt sein
        
        dt_stable_max = 0.5 / (term_x + term_y)
        return dt_stable_max


    def reset_parameters(self):
        defaults = { 
            "α_x": "0.01", "α_y": "0.01", "Nx": "50", "Ny": "50", 
            "T_Spitze": "100.0", "T_Rand": "0.0", "Zeit": "1.0", "dt": "0.001", 
            "Quelle1X": "0.5", "Quelle1Y": "0.5", "Quelle1R": "0.1", "Quelle1S": "10.0", 
            "Quelle2X": "0.25", "Quelle2Y": "0.75", "Quelle2R": "0.05", "Quelle2S": "-20.0"
        }
        for key, entry_widget in self.entries.items(): 
             is_disabled = entry_widget.cget('state') == tk.DISABLED
             if is_disabled: entry_widget.config(state=tk.NORMAL)
             entry_widget.delete(0, tk.END)
             entry_widget.insert(0, defaults[key])
             if is_disabled: entry_widget.config(state=is_disabled) 

        self.solver_var.set("Explicit FTCS"); self.bc_var.set("Neumann (Insulated)")
        self.vis_var.set("2D Heatmap (imshow)"); 
        self.source1_enabled_var.set(False) 
        self.source2_enabled_var.set(False) 
        self.laplacian_method_var.set("Numba")
        
        self.T0 = float(defaults["T_Spitze"])

        self._toggle_source1_widgets() 
        self._toggle_source2_widgets() 
        self._on_bc_change(); self._on_solver_change(); self._on_vis_change()
        self.stop_animation()
        
        try:
            if self.ax: self.ax.clear()
            self.axs_analysis[0].clear(); self.axs_analysis[1].clear()
            self._setup_analysis_plot()
            self.canvas.draw(); self.canvas_analysis.draw()
        except Exception as e: print(f"Fehler beim Leeren der Plots während Reset: {e}")
        self.status_label.config(text="Parameter zurückgesetzt.")


    def _on_bc_change(self, event=None):
        try:
            if self.bc_var.get() == "Dirichlet (Fixed Temp)": self.entries["T_Rand"].config(state=tk.NORMAL)
            else: self.entries["T_Rand"].config(state=tk.DISABLED)
        except KeyError: pass 

    def _toggle_source1_widgets(self): 
        new_state = tk.NORMAL if self.source1_enabled_var.get() else tk.DISABLED
        for lbl, entry in self.source1_widgets.values(): 
            lbl.config(state=new_state)
            entry.config(state=new_state)

    def _toggle_source2_widgets(self): 
        new_state = tk.NORMAL if self.source2_enabled_var.get() else tk.DISABLED
        for lbl, entry in self.source2_widgets.values():
            lbl.config(state=new_state)
            entry.config(state=new_state)


    def _on_vis_change(self, event=None):
        vis_type = self.vis_var.get()
        is_3d_new = "3D Surface" in vis_type
        current_is_3d = hasattr(self.ax, 'name') and self.ax.name == '3d'
        if is_3d_new != current_is_3d or self.ax is None:
            if self.ax:
                try: self.ax.remove()
                except AttributeError: pass
            if self.cbar:
                try: self.cbar.remove()
                except Exception as e: print(f"Fehler beim Entfernen alter Colorbar: {e}")
            self.cbar = None
            if is_3d_new:
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("Temperatur (°C)")
            else:
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
            self.ax.set_title("Temperaturverteilung")
            self.canvas.draw()

    def _on_solver_change(self, event=None):
        new_state = tk.NORMAL if self.solver_var.get() == "Explicit FTCS" else tk.DISABLED
        try:
            self.rb_numba.config(state=new_state)
            self.rb_numpy.config(state=new_state)
        except AttributeError: pass

    def initialize_temperature(self, Nx, Ny, T0_init, bc_type, bc_value):
        if Nx <= 1 or Ny <= 1: raise ValueError("Gittergröße Nx und Ny müssen > 1 sein.")
        u = np.zeros((Ny, Nx)); dx = 1.0 / (Nx - 1); dy = 1.0 / (Ny - 1)
        x = np.linspace(0, 1, Nx); y = np.linspace(0, 1, Ny); X, Y = np.meshgrid(x, y)
        u = T0_init * np.exp(-80 * ((X - 0.5)**2 + (Y - 0.5)**2))
        if bc_type == "Dirichlet (Fixed Temp)":
            u[0, :] = bc_value; u[-1, :] = bc_value; u[:, 0] = bc_value; u[:, -1] = bc_value
        return u, x, y, dx, dy

    def create_source_term(self, Nx, Ny, dx, dy, 
                           enabled1, sx1, sy1, sr1, s_strength1,
                           enabled2, sx2, sy2, sr2, s_strength2):
        S_total = np.zeros((Ny, Nx))
        y_indices, x_indices = np.indices((Ny, Nx))

        # Quelle 1
        if enabled1 and sr1 > 0:
            dist_sq1 = ((x_indices * dx - sx1)**2 + (y_indices * dy - sy1)**2)
            mask1 = dist_sq1 <= sr1**2
            S_total[mask1] += s_strength1 

        # Quelle 2 (NEU)
        if enabled2 and sr2 > 0:
            dist_sq2 = ((x_indices * dx - sx2)**2 + (y_indices * dy - sy2)**2)
            mask2 = dist_sq2 <= sr2**2
            S_total[mask2] += s_strength2
            
        return S_total

    def solve_ftcs(self, u_initial, alpha_x, alpha_y, dt, dx, dy, Nx, Ny, t_max, bc_type, bc_value, S_full, laplacian_method):
        
      if Nx <= 2 or Ny <= 2: raise ValueError("Gitter für FTCS zu klein (Nx/Ny > 2 benötigt).")
        u_current = u_initial.copy(); frames = [u_current.copy()]; analysis_t = [0.0]
        analysis_max_T = [np.max(u_current)]; analysis_total_H = [np.sum(u_current) * dx * dy]
        rx = alpha_x * dt / dx**2 if dx > 1e-12 else 0
        ry = alpha_y * dt / dy**2 if dy > 1e-12 else 0
        stability_factor = rx + ry
        if stability_factor > 0.5 + 1e-9:
             stable_dt_limit = self._calculate_stable_dt(alpha_x, alpha_y, Nx, Ny)
             error_msg = f"Instabilität! (rx+ry={stability_factor:.4f}>0.5). Stabiles dt <= {stable_dt_limit:.3e} (aktuell: {dt:.3e}) benötigt."
             raise ValueError(error_msg)
        num_steps = int(round(t_max / dt)) if dt > 1e-12 else 0
        print(f"FTCS ({laplacian_method}): Starte {num_steps} Schritte (rx={rx:.3f}, ry={ry:.3f})")
        inv_dx2 = 1.0 / dx**2 if dx > 1e-12 else 0; inv_dy2 = 1.0 / dy**2 if dy > 1e-12 else 0
        i_int_slice, j_int_slice = slice(1, Ny-1), slice(1, Nx-1)
        S_interior = S_full[i_int_slice, j_int_slice]
        Ny_int, Nx_int = Ny - 2, Nx - 2
        compile_time = 0
        if laplacian_method == "Numba" and num_steps > 0:
            t_start_compile = time.time()
            _ = ftcs_step_numba(u_current, S_interior, alpha_x, alpha_y, dt, inv_dx2, inv_dy2, Ny_int, Nx_int)
            compile_time = time.time() - t_start_compile
            print(f"  Numba Kompilierzeit: {compile_time:.4f}s")
        t_start_loop = time.time()
        for n in range(num_steps):
            u_new = u_current.copy()
            if laplacian_method == "Numba":
                u_new_interior = ftcs_step_numba(u_current, S_interior, alpha_x, alpha_y, dt, inv_dx2, inv_dy2, Ny_int, Nx_int)
            else:
                u_new_interior = ftcs_step_numpy(u_current, S_interior, alpha_x, alpha_y, dt, dx, dy, Ny_int, Nx_int)
            u_new[i_int_slice, j_int_slice] = u_new_interior
            if bc_type == "Dirichlet (Fixed Temp)":
                u_new[0, :] = bc_value; u_new[-1, :] = bc_value; u_new[:, 0] = bc_value; u_new[:, -1] = bc_value
            elif bc_type == "Neumann (Insulated)":
                 u_new[0, :] = u_new[1, :]; u_new[-1, :] = u_new[-2, :]
                 u_new[:, 0] = u_new[:, 1]; u_new[:, -1] = u_new[:, -2]
                 u_new[0,0] = u_new[1,1]; u_new[0,-1] = u_new[1,-2]; u_new[-1,0] = u_new[-2,1]; u_new[-1,-1] = u_new[-2,-2]
            u_current = u_new
            save_interval = max(1, num_steps // 100) if num_steps > 0 else 1
            if n % save_interval == 0 or n == num_steps - 1:
                frames.append(u_current.copy()); current_time = (n + 1) * dt
                analysis_t.append(current_time); analysis_max_T.append(np.max(u_current)); analysis_total_H.append(np.sum(u_current) * dx * dy)
        loop_time = time.time() - t_start_loop
        if num_steps > 0 : print(f"FTCS ({laplacian_method}): Schleife beendet in {loop_time:.4f}s.")
        return frames, analysis_t, analysis_max_T, analysis_total_H

    def solve_crank_nicolson(self, u_initial, alpha_x, alpha_y, dt, dx, dy, Nx, Ny, t_max, bc_type, bc_value, S_full):
        # ... (Inhalt bleibt im Wesentlichen gleich, S_full wird verwendet) ...
        if Nx <= 2 or Ny <= 2: raise ValueError("Gitter für CN zu klein (Nx/Ny > 2 benötigt).")
        u_current = u_initial.copy(); frames = [u_current.copy()]; analysis_t = [0.0]
        analysis_max_T = [np.max(u_current)]; analysis_total_H = [np.sum(u_current) * dx * dy]
        rx = alpha_x * dt / (2.0 * dx**2) if dx > 1e-12 else 0; ry = alpha_y * dt / (2.0 * dy**2) if dy > 1e-12 else 0
        num_steps = int(round(t_max / dt)) if dt > 1e-12 else 0
        print(f"Crank-Nicolson: Starte {num_steps} Schritte (eff. rx={rx*2:.3f}, ry={ry*2:.3f})")
        N_int = (Nx - 2) * (Ny - 2)
        if N_int <= 0: return frames, analysis_t, analysis_max_T, analysis_total_H
        main_diag_val = 1 + 2*rx + 2*ry
        A_diagonals = [np.full(N_int, main_diag_val)]
        A_offsets = [0]
        if N_int - 1 > 0: # Für Off-Diagonalen x
            diag_x = np.full(N_int - 1, -rx)
            diag_x[(Nx-2)-1::(Nx-2)] = 0 # Zeilenumbrüche nullen
            A_diagonals.extend([diag_x, diag_x])
            A_offsets.extend([-1, 1])
        if N_int - (Nx-2) > 0: # Für Off-Diagonalen y
            diag_y = np.full(N_int - (Nx-2), -ry)
            A_diagonals.extend([diag_y, diag_y])
            A_offsets.extend([-(Nx-2), (Nx-2)])
        A = diags(A_diagonals, A_offsets, shape=(N_int, N_int)).tocsr()
        print("Crank-Nicolson: Matrix A erstellt.")
        S_interior = S_full[1:-1, 1:-1]
        t_start_loop = time.time()
        for n in range(num_steps):
            u_int_n = u_current[1:-1, 1:-1]
            Lu_n_x = rx * (u_current[1:-1, :-2] - 2*u_int_n + u_current[1:-1, 2:]) if dx > 1e-12 else 0
            Lu_n_y = ry * (u_current[:-2, 1:-1] - 2*u_int_n + u_current[2:, 1:-1]) if dy > 1e-12 else 0
            term_b = u_int_n + Lu_n_x + Lu_n_y
            source_term_explicit = dt * S_interior
            b_flat = (term_b + source_term_explicit).flatten()
            if bc_type == "Dirichlet (Fixed Temp)":
                val_rx_bc = 2 * rx * bc_value; val_ry_bc = 2 * ry * bc_value
                for i_int_row in range(Ny - 2):
                    b_flat[i_int_row * (Nx - 2)] += val_rx_bc
                    b_flat[i_int_row * (Nx - 2) + (Nx - 3)] += val_rx_bc
                for j_int_col in range(Nx - 2):
                    b_flat[j_int_col] += val_ry_bc
                    b_flat[(Ny - 3) * (Nx - 2) + j_int_col] += val_ry_bc
            try:
                u_new_flat = spsolve(A, b_flat)
                if np.any(np.isnan(u_new_flat)) or np.any(np.isinf(u_new_flat)): raise ValueError("NaN oder Inf im Solver-Ergebnis CN!")
            except Exception as e:
                print(f"Solver Fehler Schritt {n+1} (CN): {e}"); messagebox.showerror("Solver Fehler CN", f"LGS konnte nicht gelöst werden:\n{e}")
                return frames, analysis_t, analysis_max_T, analysis_total_H
            u_current[1:-1, 1:-1] = u_new_flat.reshape((Ny - 2, Nx - 2))
            if bc_type == "Dirichlet (Fixed Temp)":
                u_current[0, :] = bc_value; u_current[-1, :] = bc_value; u_current[:, 0] = bc_value; u_current[:, -1] = bc_value
            elif bc_type == "Neumann (Insulated)":
                 u_current[0, :] = u_current[1, :]; u_current[-1, :] = u_current[-2, :]
                 u_current[:, 0] = u_current[:, 1]; u_current[:, -1] = u_current[:, -2]
                 u_current[0,0] = u_current[1,1]; u_current[0,-1] = u_current[1,-2]; u_current[-1,0] = u_current[-2,1]; u_current[-1,-1] = u_current[-2,-2]
            save_interval = max(1, num_steps // 100) if num_steps > 0 else 1
            if n % save_interval == 0 or n == num_steps - 1:
                frames.append(u_current.copy()); current_time = (n + 1) * dt
                analysis_t.append(current_time); analysis_max_T.append(np.max(u_current)); analysis_total_H.append(np.sum(u_current) * dx * dy)
        loop_time = time.time() - t_start_loop
        if num_steps > 0: print(f"Crank-Nicolson: Schleife beendet in {loop_time:.4f}s.")
        return frames, analysis_t, analysis_max_T, analysis_total_H


    def start_simulation(self):
        self.stop_animation()
        try:
            alpha_x = float(self.entries["α_x"].get()); alpha_y = float(self.entries["α_y"].get())
            Nx = int(self.entries["Nx"].get()); Ny = int(self.entries["Ny"].get())
            t_max = float(self.entries["Zeit"].get()); dt = float(self.entries["dt"].get())
            T0_param = float(self.entries["T_Spitze"].get())
            
            bc_type = self.bc_var.get()
            bc_value = 0.0
            if bc_type == "Dirichlet (Fixed Temp)": bc_value = float(self.entries["T_Rand"].get())
            
            # Quelle 1 Parameter
            source1_enabled = self.source1_enabled_var.get()
            sx1 = float(self.entries["Quelle1X"].get()) if source1_enabled else 0.0
            sy1 = float(self.entries["Quelle1Y"].get()) if source1_enabled else 0.0
            sr1 = float(self.entries["Quelle1R"].get()) if source1_enabled else 0.0
            s_strength1 = float(self.entries["Quelle1S"].get()) if source1_enabled else 0.0
            
            # Quelle 2 Parameter 
            source2_enabled = self.source2_enabled_var.get()
            sx2 = float(self.entries["Quelle2X"].get()) if source2_enabled else 0.0
            sy2 = float(self.entries["Quelle2Y"].get()) if source2_enabled else 0.0
            sr2 = float(self.entries["Quelle2R"].get()) if source2_enabled else 0.0
            s_strength2 = float(self.entries["Quelle2S"].get()) if source2_enabled else 0.0

            # Validierungen
            if Nx <= 2 or Ny <= 2: raise ValueError("Nx und Ny müssen größer als 2 sein.")
            if dt <= 0: raise ValueError("Zeitschritt dt muss positiv sein.")
            if t_max <= 0: raise ValueError("Simulationsdauer muss positiv sein.")
            if alpha_x < 0 or alpha_y < 0: raise ValueError("Wärmeleitfähigkeit (α) darf nicht negativ sein.")
            if source1_enabled and sr1 < 0: raise ValueError("Quellradius 1 darf nicht negativ sein.")
            if source2_enabled and sr2 < 0: raise ValueError("Quellradius 2 darf nicht negativ sein.")


            self.T0 = T0_param
            self.temp_field, self.grid_x, self.grid_y, dx, dy = self.initialize_temperature(Nx, Ny, self.T0, bc_type, bc_value)
            
            S_full = self.create_source_term(Nx, Ny, dx, dy, 
                                             source1_enabled, sx1, sy1, sr1, s_strength1,
                                             source2_enabled, sx2, sy2, sr2, s_strength2)
            
            solver_choice = self.solver_var.get()
            laplacian_method_choice = self.laplacian_method_var.get()

            status_text = f"Starte ({solver_choice}"
            if solver_choice == "Explicit FTCS": status_text += f" / {laplacian_method_choice}"
            status_text += ")..."
            self.status_label.config(text=status_text); self.master.update_idletasks()

            if solver_choice == "Explicit FTCS":
                 frames, t, max_T, total_H = self.solve_ftcs(self.temp_field.copy(), alpha_x, alpha_y, dt, dx, dy, Nx, Ny, t_max, bc_type, bc_value, S_full, laplacian_method_choice)
            elif solver_choice == "Implicit CN":
                 frames, t, max_T, total_H = self.solve_crank_nicolson(self.temp_field.copy(), alpha_x, alpha_y, dt, dx, dy, Nx, Ny, t_max, bc_type, bc_value, S_full)
            else: raise ValueError("Unbekannter Solver ausgewählt.")

            if not frames: self.status_label.config(text="Simulation fehlgeschlagen (Solver)."); return

            self.analysis_data = {'t': t, 'max_T': max_T, 'total_H': total_H}
            self.setup_and_start_animation(frames, S_full, source1_enabled or source2_enabled) # any source active for scatter
            self.update_analysis_plot()
            self.status_label.config(text="Simulation abgeschlossen. Animation läuft.")
            self.stop_button.config(state=tk.NORMAL)

        except ValueError as e: messagebox.showerror("Parameterfehler", f"Ungültige Eingabe: {e}"); self.status_label.config(text="Fehlerhafte Eingabe.")
        except Exception as e: messagebox.showerror("Simulationsfehler", f"Ein Fehler ist aufgetreten:\n{e}"); self.status_label.config(text="Simulationsfehler."); import traceback; traceback.print_exc()

    def stop_animation(self):
         if self.anim:
            try: self.anim.event_source.stop(); self.anim = None; self.stop_button.config(state=tk.DISABLED); self.start_button.config(state=tk.NORMAL); self.status_label.config(text="Animation gestoppt.")
            except AttributeError: self.anim = None; self.stop_button.config(state=tk.DISABLED)

    def setup_and_start_animation(self, frames, S_full, any_source_enabled): # Parameter any_source_enabled
        if self.ax:
            try: self.ax.remove()
            except AttributeError: pass
        self.ax = None
        if self.cbar:
            try: self.cbar.remove()
            except Exception: pass
        self.cbar = None

        vis_type = self.vis_var.get()
        is_3d_new = "3D Surface" in vis_type
        if is_3d_new:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("Temp (°C)")
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        
        all_frames_arr = np.array(frames)
        data_min_val = np.min(all_frames_arr)
        data_max_val = np.max(all_frames_arr)

        if self.T0 is not None:
            data_min_val = min(data_min_val, self.T0)
            data_max_val = max(data_max_val, self.T0)
        if abs(data_max_val - data_min_val) < 1e-6: data_min_val -= 0.5; data_max_val += 0.5
        
        
        current_cmap = 'jet' 

        mappable = None 
        if is_3d_new:
            X, Y = np.meshgrid(self.grid_x, self.grid_y)
            rstride_val = max(1, len(self.grid_y) // 50); cstride_val = max(1, len(self.grid_x) // 50)
            self.surf = self.ax.plot_surface(X, Y, frames[0], cmap=current_cmap, vmin=data_min_val, vmax=data_max_val, rstride=rstride_val, cstride=cstride_val, edgecolor='none', antialiased=False)
            self.ax.set_zlim(data_min_val - 0.1*(data_max_val-data_min_val), data_max_val + 0.1*(data_max_val-data_min_val))
            mappable = self.surf
            fargs_anim = (frames, X, Y, is_3d_new, data_min_val, data_max_val)
            update_func = self._update_plot_3d
        else: 
            self.img = self.ax.imshow(frames[0], cmap=current_cmap, origin='lower', vmin=data_min_val, vmax=data_max_val, extent=[self.grid_x[0], self.grid_x[-1], self.grid_y[0], self.grid_y[-1]], interpolation='bilinear', aspect='auto')
            mappable = self.img
            if self.scatter_source:
                try: self.scatter_source.remove()
                except Exception: pass
            self.scatter_source = None
            if any_source_enabled: 
                source_mask = S_full != 0
                if np.any(source_mask):
                     y_idx, x_idx = np.where(source_mask)
                     source_colors = ['cyan' if S_full[y,x]>0 else 'magenta' for y,x in zip(y_idx, x_idx)]
                     self.scatter_source = self.ax.scatter(self.grid_x[x_idx], self.grid_y[y_idx], s=20, c=source_colors, marker='o', edgecolors='black', linewidth=0.5, label='Quelle/Senke', alpha=0.7, zorder=5)
                     if self.scatter_source and not self.ax.get_legend(): self.ax.legend(handles=[self.scatter_source], fontsize='small', loc='upper right')
            fargs_anim = (frames, is_3d_new)
            update_func = self._update_plot_2d
        
        if mappable: self.cbar = self.fig.colorbar(mappable, ax=self.ax, shrink=0.8, aspect=15, label="Temperatur (°C)")
        if self.analysis_data['t']: # Sicherstellen, dass Zeitdaten vorhanden sind
             self.ax.set_title(f"Wärmeverteilung T(x,y,t={self.analysis_data['t'][0]:.3f}s)")
        else:
             self.ax.set_title("Wärmeverteilung T(x,y,t)")

        self.anim = FuncAnimation(self.fig, update_func, frames=len(frames), fargs=fargs_anim, interval=50, blit=False, repeat=False)
        self.canvas.draw()

    def _update_plot_2d(self, frame_idx, frames, is_3d): 
        if not self.anim: return []
        self.img.set_array(frames[frame_idx])
        if frame_idx < len(self.analysis_data['t']): actual_time = self.analysis_data['t'][frame_idx]; self.ax.set_title(f"T(x, y, t={actual_time:.3f}s)")
        return [self.img]

    def _update_plot_3d(self, frame_idx, frames, X, Y, is_3d, data_min_val, data_max_val):
        if not self.anim: return []
        current_elev, current_azim = self.ax.elev, self.ax.azim
        self.ax.clear() 
        new_z_data = frames[frame_idx]
        rstride_val = max(1, Y.shape[0] // 50); cstride_val = max(1, X.shape[1] // 50)

        current_cmap = 'jet' # oder 'coolwarm'
        self.surf = self.ax.plot_surface(X, Y, new_z_data, cmap=current_cmap, vmin=data_min_val, vmax=data_max_val, rstride=rstride_val, cstride=cstride_val, edgecolor='none', antialiased=False)
        self.ax.set_zlim(data_min_val - 0.1*(data_max_val-data_min_val), data_max_val + 0.1*(data_max_val-data_min_val))
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("Temp (°C)")
        self.ax.view_init(elev=current_elev, azim=current_azim)
        if frame_idx < len(self.analysis_data['t']): actual_time = self.analysis_data['t'][frame_idx]; self.ax.set_title(f"T(x, y, t={actual_time:.3f}s)")
        else: self.ax.set_title("Wärmeverteilung T(x, y, t)")
        return []

    def update_analysis_plot(self):
        t = self.analysis_data['t']; max_T = self.analysis_data['max_T']; total_H = self.analysis_data['total_H']
        if not t: # Wenn keine Daten, nichts tun
            self.axs_analysis[0].clear(); self.axs_analysis[1].clear()
            self._setup_analysis_plot() # Labels etc. wiederherstellen
            self.canvas_analysis.draw()
            return

        self.line_max_T.set_data(t, max_T); self.line_total_H.set_data(t, total_H)
        for ax_an in self.axs_analysis: ax_an.relim(); ax_an.autoscale_view()
        if total_H and all(isinstance(h, (int, float)) and not math.isnan(h) and not math.isinf(h) for h in total_H):
            min_h = min(total_H) if total_H else 0; current_ylim_bottom = self.axs_analysis[1].get_ylim()[0]
            self.axs_analysis[1].set_ylim(bottom=min(0, min_h - 0.1 * abs(min_h) if min_h !=0 else -0.1, current_ylim_bottom))
        self.canvas_analysis.draw()

# --- Hauptprogramm ---
if __name__ == "__main__":
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except ImportError: pass
    root = tk.Tk()
    style = ttk.Style()
    try:
        available_themes = style.theme_names(); preferred_themes = ['clam', 'alt', 'vista', 'xpnative', 'aqua', 'default']
        for theme in preferred_themes:
            if theme in available_themes:
                try: style.theme_use(theme); break
                except tk.TclError: pass
    except tk.TclError: print("Konnte ttk Themes nicht prüfen.")
    app = AdvancedHeatEquationGUI(root)
    root.mainloop()
    print("Anwendung beendet.")
