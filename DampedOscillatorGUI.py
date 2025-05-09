import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation 
import math 

class DampedOscillatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gedämpfte Schwingung: Simulation & Analyse")
        # self.root.geometry("950x650") # Größeres Fenster

        self.anim = None # Variable zum Speichern des Animationsobjekts
        self.sol = None # Variable zum Speichern der ODE-Lösung
        self.line = None # Variable für die animierte Linie

        # --- Hauptlayout erstellen (Links: Steuerung+Plot, Rechts: Infos) ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_frame = ttk.LabelFrame(main_frame, text="Analyse & Formeln", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        # --- Linker Frame: Eingaben und Plot ---
        input_frame = ttk.LabelFrame(left_frame, text="Parameter", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        plot_button_frame = ttk.Frame(left_frame)
        plot_button_frame.pack(fill=tk.X, pady=(0,10))

        plot_canvas_frame = ttk.Frame(left_frame)
        plot_canvas_frame.pack(fill=tk.BOTH, expand=True)


        # Parameter Eingabefelder in input_frame
        self.create_input_fields(input_frame)

        # Simulations-Button in plot_button_frame
        self.plot_button = ttk.Button(plot_button_frame, text="Simulation starten / neu starten", command=self.run_simulation)
        self.plot_button.pack(pady=5)

        # Matplotlib Figure in plot_canvas_frame
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Rechter Frame: Formeln und Berechnungen ---
        self.create_info_display(right_frame)

        # --- Initialisierung ---
        self._update_info_display() # Zeige initiale Formeln etc.
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)


    def create_input_fields(self, parent_frame):
        """Erstellt Labels und Eingabefelder für Parameter."""
        labels = ["Masse (m, kg):", "Dämpfung (b, Ns/m):", "Federkonstante (k, N/m):",
                  "Anfangsauslenkung (x0, m):", "Anfangsgeschw. (v0, m/s):", "Simulationsdauer (t_end, s):"]
        defaults = ["1.0", "0.5", "10.0", "1.0", "0.0", "20.0"] # Angepasste Defaults
        self.entries = {}

        for i, label_text in enumerate(labels):
            ttk.Label(parent_frame, text=label_text).grid(row=i, column=0, padx=5, pady=3, sticky="w")
            entry = ttk.Entry(parent_frame, width=10)
            entry.grid(row=i, column=1, padx=5, pady=3, sticky="we")
            entry.insert(0, defaults[i])
            # Schlüssel vereinfacht (ohne Einheiten)
            key = label_text.split(" ")[0]
            self.entries[key] = entry

        # Grid-Konfiguration für gleichmäßige Verteilung im parent_frame
        parent_frame.columnconfigure(1, weight=1)


    def create_info_display(self, parent_frame):
        """Erstellt Labels zur Anzeige von Formeln und Berechnungen."""
        ttk.Label(parent_frame, text="Bewegungsgleichung:", font="-weight bold").pack(anchor=tk.W, pady=(0,2))
        # Wir verwenden hier einfachen Text mit Unicode, da echtes LaTeX in Tkinter komplex ist.
        # Besser: Formel im Plot anzeigen (siehe _update_plot_text)
        self.eq_label = ttk.Label(parent_frame, text="m⋅ẍ + b⋅ẋ + k⋅x = 0", font=("Arial", 11))
        self.eq_label.pack(anchor=tk.W, padx=5, pady=(0,10))

        ttk.Label(parent_frame, text="Wichtige Parameter:", font="-weight bold").pack(anchor=tk.W, pady=(5,2))

        self.omega0_label = ttk.Label(parent_frame, text="Eigenkreisfrequenz (ω₀): - rad/s")
        self.omega0_label.pack(anchor=tk.W, padx=5)
        self.zeta_label = ttk.Label(parent_frame, text="Dämpfungsgrad (ζ): - ")
        self.zeta_label.pack(anchor=tk.W, padx=5)
        self.case_label = ttk.Label(parent_frame, text="Fall: -")
        self.case_label.pack(anchor=tk.W, padx=5, pady=(0,10))

        self.omega_d_label = ttk.Label(parent_frame, text="Ged. Kreisfrequenz (ω_d): - rad/s")
        self.omega_d_label.pack(anchor=tk.W, padx=5)
        self.period_label = ttk.Label(parent_frame, text="Periode (T_d): - s")
        self.period_label.pack(anchor=tk.W, padx=5)

        ttk.Label(parent_frame, text="\nFormeln:", font="-weight bold").pack(anchor=tk.W, pady=(10,2))
        # Unicode-Version der Formeln
        formulas = [
            "ω₀ = √(k/m)",
            "ζ = b / (2⋅√(m⋅k))",
            "Schwingfall (ζ < 1):",
            "  ω_d = ω₀⋅√(1-ζ²)",
            "  T_d = 2π / ω_d",
            "Kriechfall (ζ > 1)",
            "Grenzfall (ζ = 1)",
        ]
        for f in formulas:
            ttk.Label(parent_frame, text=f, font=("Arial", 9)).pack(anchor=tk.W, padx=5)


    def _update_info_display(self, m=None, b=None, k=None):
        """Aktualisiert die Labels mit berechneten Werten."""
        if m is None or b is None or k is None or m <= 0 or k <= 0:
            self.omega0_label.config(text="Eigenkreisfrequenz (ω₀): - rad/s")
            self.zeta_label.config(text="Dämpfungsgrad (ζ): -")
            self.case_label.config(text="Fall: -")
            self.omega_d_label.config(text="Ged. Kreisfrequenz (ω_d): - rad/s")
            self.period_label.config(text="Periode (T_d): - s")
            return

        try:
            # Berechnungen
            omega0 = math.sqrt(k / m)
            zeta = b / (2 * math.sqrt(m * k))

            case = "Ungedämpft"
            omega_d = omega0
            period = math.inf

            if b > 1e-9: # Nur wenn Dämpfung vorhanden ist
                if abs(zeta - 1) < 1e-9: # Numerische Toleranz für Gleichheit
                    case = "Aperiodischer Grenzfall (ζ = 1)"
                    omega_d = 0
                    period = math.inf
                elif zeta > 1:
                    case = "Kriechfall (Überdämpft) (ζ > 1)"
                    omega_d = 0 # Keine Oszillation
                    period = math.inf
                else: # zeta < 1
                    case = "Schwingfall (Unterdämpft) (ζ < 1)"
                    omega_d = omega0 * math.sqrt(1 - zeta**2)
                    if omega_d > 1e-9:
                         period = 2 * math.pi / omega_d
                    else: # Sehr starke Dämpfung, fast Grenzfall
                         period = math.inf
            else: # Ungedämpft b=0 -> zeta=0
                 zeta = 0.0
                 period = 2 * math.pi / omega0 if omega0 > 1e-9 else math.inf


            # Labels aktualisieren
            self.omega0_label.config(text=f"Eigenkreisfrequenz (ω₀): {omega0:.3f} rad/s")
            self.zeta_label.config(text=f"Dämpfungsgrad (ζ): {zeta:.3f}")
            self.case_label.config(text=f"Fall: {case}")

            if case == "Schwingfall (Unterdämpft) (ζ < 1)" and omega_d > 1e-9:
                self.omega_d_label.config(text=f"Ged. Kreisfrequenz (ω_d): {omega_d:.3f} rad/s")
                self.period_label.config(text=f"Periode (T_d): {period:.3f} s")
            else:
                self.omega_d_label.config(text="Ged. Kreisfrequenz (ω_d): - (nicht oszillierend)")
                self.period_label.config(text="Periode (T_d): -")

        except ValueError: # z.B. bei negativer Wurzel
            messagebox.showerror("Fehler", "Ungültige Parameter für Berechnungen (z.B. m oder k <= 0).")
        except Exception as e:
             messagebox.showerror("Fehler", f"Unbekannter Fehler bei Berechnungen: {e}")


    def damped_oscillator_ode(self, t, y, m, b, k):
        """Das System von DGLs für solve_ivp."""
        x, v = y
        # Sicherstellen, dass m nicht null ist
        if abs(m) < 1e-12: m = 1e-12 # Minimalwert, um Division durch Null zu vermeiden
        dx_dt = v
        dv_dt = -(b/m) * v - (k/m) * x
        return [dx_dt, dv_dt]


    def run_simulation(self):
        """Liest Parameter, löst die DGL und startet die Animation."""
        # Laufende Animation stoppen, falls vorhanden
        if self.anim is not None:
            # self.anim.event_source.stop() # Sicherer Weg zum Stoppen?
            # Workaround: Animationssteuerung über Flag oder None setzen
             self.anim = None # Signalisiert _update_animation aufzuhören
             plt.close(self.fig) # Schließe alte Figur komplett
             # Erstelle Figur und Achse neu
             self.fig, self.ax = plt.subplots(figsize=(7, 5))
             # Bette neue Figur wieder ein
             # Entferne altes Canvas-Widget, falls vorhanden
             if hasattr(self, 'canvas_widget'): self.canvas_widget.pack_forget()
             self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_widget.master) # Nutze den alten Master-Frame
             self.canvas_widget = self.canvas.get_tk_widget()
             self.canvas_widget.pack(fill=tk.BOTH, expand=True)


        try:
            # Parameter aus Eingabefeldern lesen
            m = float(self.entries["Masse"].get())
            b = float(self.entries["Dämpfung"].get())
            k = float(self.entries["Federkonstante"].get())
            x0 = float(self.entries["Anfangsauslenkung"].get())
            v0 = float(self.entries["Anfangsgeschw."].get())
            t_end = float(self.entries["Simulationsdauer"].get())

            if m <= 0 or k < 0 or b < 0 or t_end <= 0:
                messagebox.showerror("Fehler", "Masse und Simulationsdauer müssen positiv sein. k und b dürfen nicht negativ sein.")
                return

            # Berechnungen durchführen und Anzeige aktualisieren
            self._update_info_display(m, b, k)

            # Zeitspanne und Auswertungspunkte
            t_span = (0, t_end)
            # Mehr Punkte für flüssigere Animation (aber nicht zu viele)
            num_frames = int(t_end * 50) + 1 # z.B. 50 Frames pro Sekunde
            t_eval = np.linspace(t_span[0], t_span[1], num_frames)

            # Anfangswerte
            y0 = [x0, v0]

            # ODE lösen
            print("Löse ODE...")
            self.sol = solve_ivp(self.damped_oscillator_ode, t_span, y0, args=(m, b, k), t_eval=t_eval, dense_output=True)
            print("ODE gelöst.")

            # Plot vorbereiten (Achsen, Titel etc.)
            self.ax.clear()
            # Plot-Grenzen dynamisch setzen (etwas über Max/Min)
            max_abs_x = max(abs(np.min(self.sol.y[0])), abs(np.max(self.sol.y[0])))
            plot_limit = max_abs_x * 1.2 if max_abs_x > 0.1 else 0.5 # Mindestlimit
            self.ax.set_ylim(-plot_limit, plot_limit)
            self.ax.set_xlim(t_span)

            self.ax.set_xlabel("Zeit (s)")
            self.ax.set_ylabel("Auslenkung x(t) (m)")
            self.ax.set_title("Gedämpfte Schwingung (Animation)")
            self.ax.grid(True)

            # Formel auf Plot anzeigen (Matplotlib mathtext)
            # \ddot{x} und \dot{x} für die Ableitungen
            formula_latex = r'$m \ddot{x}(t) + b \dot{x}(t) + k x(t) = 0$'
            # Positioniere den Text oben auf der Figur
            self.fig.suptitle(formula_latex, fontsize=12, y=0.99)


            # Linie für Animation initialisieren (leer)
            self.line, = self.ax.plot([], [], 'b-', lw=2, label="Auslenkung x(t)")
            self.ax.legend(loc='upper right')

            # Animation starten
            print("Starte Animation...")
            # interval anpassen für Geschwindigkeit (ms)
            # blit=False ist oft stabiler mit Tkinter
            self.anim = FuncAnimation(self.fig, self._update_animation, frames=len(t_eval),
                                      init_func=self._init_animation, interval=20, blit=False, repeat=False)

            self.canvas.draw() # Wichtig, um die initial leere Figur anzuzeigen
            print("Animation läuft.")

        except ValueError:
            messagebox.showerror("Eingabefehler", "Bitte gültige numerische Werte für alle Parameter eingeben!")
        except Exception as e:
             messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten:\n{e}")
             print(f"Fehler während Simulation/Plot: {e}")


    def _init_animation(self):
        """Initialisiert die Animationslinie."""
        if self.line is None: # Nur wenn Linie existiert
            self.line, = self.ax.plot([], [], 'b-', lw=2, label="Auslenkung x(t)") # Neu erstellen falls nötig
        self.line.set_data([], [])
        return self.line,

    def _update_animation(self, frame_index):
        """Aktualisiert die Linie für jeden Frame der Animation."""
        if self.sol is None or self.anim is None: # Prüfen ob Simulation/Animation noch läuft
            return self.line,

        try:
            # Daten bis zum aktuellen Frame holen
            current_t = self.sol.t[:frame_index+1]
            current_x = self.sol.y[0,:frame_index+1]

            # Liniendaten aktualisieren
            self.line.set_data(current_t, current_x)

            if hasattr(self, 'marker'): self.marker.remove() # Alten Marker entfernen
            self.marker, = self.ax.plot(current_t[-1], current_x[-1], 'ro') # Roter Punkt am Ende

           

            return self.line, # Komma ist wichtig für blitting (auch wenn blit=False)

        except IndexError:
             print(f"Indexfehler im Frame {frame_index}. Stoppe evtl.")
             # Animation stoppen, wenn Index ungültig wird
             if self.anim and self.anim.event_source: self.anim.event_source.stop()
             self.anim = None
             return self.line,
        except Exception as e:
            print(f"Fehler im Update Frame {frame_index}: {e}")
            # Bei Fehler Animation stoppen
            if self.anim and self.anim.event_source: self.anim.event_source.stop()
            self.anim = None
            return self.line,


    def _on_closing(self):
        """Aufräumen beim Schließen."""
        print("Schließe Anwendung...")
        # Animation stoppen, falls sie läuft
        if self.anim is not None and self.anim.event_source:
            self.anim.event_source.stop()
            self.anim = None 
        plt.close(self.fig) 
        self.root.quit()    
        self.root.destroy() # Fenster zerstören

# --- Hauptprogramm ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DampedOscillatorGUI(root)
    root.mainloop()
    print("Anwendung beendet.")
