import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading # For running simulation in a separate thread to keep GUI responsive

# --- Simulation Code (adapted from your script) ---
# Global variables for simulation results to be accessed by plotting function
sim_X, sim_Y, sim_p_next, sim_u_next, sim_v_next = None, None, None, None, None
simulation_running = False

def run_simulation(params, status_callback, progress_callback, completion_callback):
    global sim_X, sim_Y, sim_p_next, sim_u_next, sim_v_next, simulation_running
    simulation_running = True

    try:
        N_POINTS = params["N_POINTS"]
        DOMAIN_SIZE = params["DOMAIN_SIZE"]
        N_ITERATIONS = params["N_ITERATIONS"]
        TIME_STEP_LENGTH = params["TIME_STEP_LENGTH"]
        KINEMATIC_VISCOSITY = params["KINEMATIC_VISCOSITY"]
        DENSITY = params["DENSITY"]
        HORIZONTAL_VELOCITY_TOP = params["HORIZONTAL_VELOCITY_TOP"]
        N_PRESSURE_POISSON_ITERATIONS = params["N_PRESSURE_POISSON_ITERATIONS"]
        STABILITY_SAFETY_FACTOR = params["STABILITY_SAFETY_FACTOR"]

        status_callback("Initializing simulation...\n")
        element_length = DOMAIN_SIZE / (N_POINTS - 1)
        x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
        y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

        sim_X, sim_Y = np.meshgrid(x, y)

        u_prev = np.zeros_like(sim_X)
        v_prev = np.zeros_like(sim_X)
        p_prev = np.zeros_like(sim_X)

        def central_difference_x(f):
            diff = np.zeros_like(f)
            diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * element_length)
            return diff

        def central_difference_y(f):
            diff = np.zeros_like(f)
            diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * element_length)
            return diff

        def laplace(f):
            diff = np.zeros_like(f)
            diff[1:-1, 1:-1] = (f[1:-1, 0:-2] + f[0:-2, 1:-1] - 4 * f[1:-1, 1:-1] +
                                f[1:-1, 2:] + f[2:, 1:-1]) / (element_length**2)
            return diff

        maximum_possible_time_step_length = (
            0.5 * element_length**2 / KINEMATIC_VISCOSITY
        )
        if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:
            status_callback(
                f"WARNING: Stability is not guaranteed! "
                f"TIME_STEP_LENGTH ({TIME_STEP_LENGTH}) > "
                f"SAFETY_FACTOR * MAX_POSSIBLE_STEP ({STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:.4e})\n"
            )
        else:
            status_callback(
                f"Stability check passed. Max allowed step: {STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:.4e}, "
                f"Current step: {TIME_STEP_LENGTH}\n"
            )


        for iter_count in range(N_ITERATIONS):
            if not simulation_running: # Allow stopping
                status_callback("Simulation stopped by user.\n")
                return

            d_u_prev__d_x = central_difference_x(u_prev)
            d_u_prev__d_y = central_difference_y(u_prev)
            d_v_prev__d_x = central_difference_x(v_prev)
            d_v_prev__d_y = central_difference_y(v_prev)
            laplace__u_prev = laplace(u_prev)
            laplace__v_prev = laplace(v_prev)

            u_tent = (u_prev + TIME_STEP_LENGTH *
                      (-(u_prev * d_u_prev__d_x + v_prev * d_u_prev__d_y) +
                       KINEMATIC_VISCOSITY * laplace__u_prev))
            v_tent = (v_prev + TIME_STEP_LENGTH *
                      (-(u_prev * d_v_prev__d_x + v_prev * d_v_prev__d_y) +
                       KINEMATIC_VISCOSITY * laplace__v_prev))

            u_tent[0, :] = 0.0
            u_tent[:, 0] = 0.0
            u_tent[:, -1] = 0.0
            u_tent[-1, :] = HORIZONTAL_VELOCITY_TOP
            v_tent[0, :] = 0.0
            v_tent[:, 0] = 0.0
            v_tent[:, -1] = 0.0
            v_tent[-1, :] = 0.0

            d_u_tent__d_x = central_difference_x(u_tent)
            d_v_tent__d_y = central_difference_y(v_tent)

            rhs = (DENSITY / TIME_STEP_LENGTH * (d_u_tent__d_x + d_v_tent__d_y))
            
            p_next_iter = np.copy(p_prev) # Use a copy for iterative updates
            for _ in range(N_PRESSURE_POISSON_ITERATIONS):
                p_temp = np.copy(p_next_iter) # Work on a temp copy for current iteration
                p_temp[1:-1, 1:-1] = 1/4 * (
                    p_next_iter[1:-1, 0:-2] + p_next_iter[0:-2, 1:-1] +
                    p_next_iter[1:-1, 2:] + p_next_iter[2:, 1:-1] -
                    element_length**2 * rhs[1:-1, 1:-1]
                )

                p_temp[:, -1] = p_temp[:, -2]
                p_temp[0, :] = p_temp[1, :]
                p_temp[:, 0] = p_temp[:, 1]
                p_temp[-1, :] = 0.0
                p_next_iter = p_temp # Update for next Poisson iteration

            p_next = p_next_iter # Final pressure from Poisson solver

            d_p_next__d_x = central_difference_x(p_next)
            d_p_next__d_y = central_difference_y(p_next)

            u_next = (u_tent - TIME_STEP_LENGTH / DENSITY * d_p_next__d_x)
            v_next = (v_tent - TIME_STEP_LENGTH / DENSITY * d_p_next__d_y)

            u_next[0, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
            u_next[-1, :] = HORIZONTAL_VELOCITY_TOP
            v_next[0, :] = 0.0
            v_next[:, 0] = 0.0
            v_next[:, -1] = 0.0
            v_next[-1, :] = 0.0

            u_prev = u_next
            v_prev = v_next
            p_prev = p_next
            
            if (iter_count + 1) % (N_ITERATIONS // 20) == 0 or iter_count == N_ITERATIONS -1 : # Update progress bar ~20 times
                 progress_callback((iter_count + 1) / N_ITERATIONS * 100)
                 status_callback(f"Iteration {iter_count + 1}/{N_ITERATIONS} completed.\n")


        sim_p_next = p_next
        sim_u_next = u_next
        sim_v_next = v_next
        status_callback("Simulation finished successfully!\n")

    except Exception as e:
        status_callback(f"Error during simulation: {str(e)}\n")
        import traceback
        status_callback(traceback.format_exc() + "\n")
    finally:
        simulation_running = False
        completion_callback()


# --- GUI Code ---
class NavierStokesGUI:
    def __init__(self, master):
        self.master = master
        master.title("Navier-Stokes Lid-Driven Cavity Solver")
        master.geometry("1600x900")
        self.colorbar = None # Initialize colorbar variable
        self.shutting_down = False 
        self.params = {
            "N_POINTS": tk.IntVar(value=41),
            "DOMAIN_SIZE": tk.DoubleVar(value=1.0),
            "N_ITERATIONS": tk.IntVar(value=500),
            "TIME_STEP_LENGTH": tk.DoubleVar(value=0.001),
            "KINEMATIC_VISCOSITY": tk.DoubleVar(value=0.1),
            "DENSITY": tk.DoubleVar(value=1.0),
            "HORIZONTAL_VELOCITY_TOP": tk.DoubleVar(value=1.0),
            "N_PRESSURE_POISSON_ITERATIONS": tk.IntVar(value=50),
            "STABILITY_SAFETY_FACTOR": tk.DoubleVar(value=0.5)
        }

        # Main layout
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls and equations
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right panel for plot and status
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=10)

        # --- Parameters ---
        params_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)

        for i, (name, var) in enumerate(self.params.items()):
            ttk.Label(params_frame, text=name + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(params_frame, textvariable=var, width=15).grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
        params_frame.columnconfigure(1, weight=1)


        # --- Equations ---
        eq_frame = ttk.LabelFrame(left_panel, text="Governing Equations", padding="10")
        eq_frame.pack(fill=tk.X, pady=10)
        
        self.eq_canvas_widget = None
        self.render_equations(eq_frame)

        # --- Controls ---
        controls_frame = ttk.Frame(left_panel, padding="10")
        controls_frame.pack(fill=tk.X, pady=5)

        self.run_button = ttk.Button(controls_frame, text="Run Simulation", command=self.start_simulation_thread)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Simulation", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(controls_frame, text="Quit", command=master.quit).pack(side=tk.RIGHT, padx=5)
        # --- Protocol Handler for Window Closing (am Ende von __init__) ---
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        # --- Plot Area ---
        plot_frame = ttk.LabelFrame(right_panel, text="Simulation Output", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(20,20))
        plt.style.use("dark_background")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.ax.set_xlim(0, self.params["DOMAIN_SIZE"].get())
        self.ax.set_ylim(0, self.params["DOMAIN_SIZE"].get())
        self.ax.set_title("Initial State - Press 'Run Simulation'")
        self.canvas.draw()


        # --- Status Area ---
        status_frame = ttk.LabelFrame(right_panel, text="Log & Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)


    def render_equations(self, parent_frame):
        # Equations as LaTeX strings
        eq_momentum = r"$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$"
        eq_incompressibility = r"$\nabla \cdot \mathbf{u} = 0$"
        
        eq_strategy_title = r"\textbf{Solution Strategy (Chorin's Projection):}"
        eq_step1 = r"1. Solve for tentative velocity $\mathbf{u}^*$:"
        eq_step1_detail = r"$\frac{\mathbf{u}^* - \mathbf{u}^n}{\Delta t} + (\mathbf{u}^n \cdot \nabla) \mathbf{u}^n = \nu \nabla^2 \mathbf{u}^*$"
        eq_step2 = r"2. Solve pressure Poisson equation for $p^{n+1}$:"
        eq_step2_detail = r"$\nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$"
        eq_step3 = r"3. Correct velocity for $\mathbf{u}^{n+1}$:"
        eq_step3_detail = r"$\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1}$"

        scenario_title = r"\textbf{Lid-Driven Cavity Scenario:}"
        scenario_desc = (
            r"- Velocity and pressure have zero initial condition." "\n"
            r"- Homogeneous Dirichlet Boundary Conditions (u=0, v=0) on 3 walls." "\n"
            r"- Top wall: $u = u_{top}$, $v=0$."
        )


        fig_eq = plt.figure(figsize=(5,4)) # Adjust size as needed
        fig_eq.patch.set_facecolor('black') # Match typical Tkinter bg
        
        ax_eq = fig_eq.add_subplot(111)
        ax_eq.axis('off')

        eq_text = (
            f"{eq_momentum}\n\n"
            f"{eq_incompressibility}\n\n"
            f"{eq_strategy_title}\n"
            f"{eq_step1}\n{eq_step1_detail}\n"
            f"{eq_step2}\n{eq_step2_detail}\n"
            f"{eq_step3}\n{eq_step3_detail}\n\n"
            f"{scenario_title}\n{scenario_desc}"
        )
        
        # Reduce font size for better fit
        ax_eq.text(0.01, 0.99, eq_text, va='top', ha='left', fontsize=9, linespacing=1.8) 
        fig_eq.tight_layout(pad=0.2)

        if self.eq_canvas_widget:
            self.eq_canvas_widget.destroy()

        eq_canvas = FigureCanvasTkAgg(fig_eq, master=parent_frame)
        self.eq_canvas_widget = eq_canvas.get_tk_widget()
        self.eq_canvas_widget.pack(fill=tk.BOTH, expand=True)
        eq_canvas.draw()
        plt.close(fig_eq) # Close the figure to free memory, canvas still holds it

    def add_status(self, message):
        self.master.after(0, self._add_status_thread_safe, message)



    def update_progress(self, value):
         self.master.after(0, self._update_progress_thread_safe, value)

    


    def simulation_complete(self):
        self.master.after(0, self._simulation_complete_thread_safe)

    def _add_status_thread_safe(self, message):
        # Nicht ausführen, wenn heruntergefahren wird oder Master/Widget nicht mehr existiert
        if self.shutting_down or \
           not (hasattr(self.master, 'winfo_exists') and self.master.winfo_exists()):
            # print(f"Status-Log (Herunterfahren/Fenster weg): {message.strip()}") # Optional: Konsolenausgabe
            return
        
        try:
            # Spezifisches Widget prüfen
            if not (hasattr(self, 'status_text') and self.status_text.winfo_exists()):
                return

            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, message)
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)
        except tk.TclError:
            # Widget wurde wahrscheinlich zerstört, während wir versuchten, es zu konfigurieren
            # print(f"Status-Log (TclError in _add_status_thread_safe): {message.strip()}") # Optional
            pass

    def _update_progress_thread_safe(self, value):
        if self.shutting_down or \
           not (hasattr(self.master, 'winfo_exists') and self.master.winfo_exists()):
            return
        try:
            if not (hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists()):
                return
            self.progress_bar['value'] = value
        except tk.TclError:
            pass

    def _simulation_complete_thread_safe(self):
        if self.shutting_down or \
           not (hasattr(self.master, 'winfo_exists') and self.master.winfo_exists()):
            return
        try:
            # Prüfen Sie jedes Widget, bevor Sie darauf zugreifen
            if hasattr(self, 'run_button') and self.run_button.winfo_exists():
                self.run_button.config(state=tk.NORMAL)
            if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                self.stop_button.config(state=tk.DISABLED)
            if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                self.progress_bar['value'] = 0
            
            global sim_X 
            if sim_X is not None:
                if hasattr(self, 'plot_results'): self.plot_results()
            else:
                # add_status prüft bereits self.shutting_down
                if hasattr(self, 'add_status'): self.add_status("Simulation did not produce results to plot.\n")
        except tk.TclError:
            pass


    def start_simulation_thread(self):
        global simulation_running
        simulation_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.add_status("Starting simulation...\n")

        current_params = {name: var.get() for name, var in self.params.items()}
        
        # Only update the title. Let plot_results handle full clearing.
        self.ax.set_title("Simulation Running...")
        # Update limits in case domain size changed and wasn't plotted yet.
        self.ax.set_xlim(0, current_params["DOMAIN_SIZE"])
        self.ax.set_ylim(0, current_params["DOMAIN_SIZE"])
        self.canvas.draw() # Redraw with new title and potentially limits

        self.sim_thread = threading.Thread(
            target=run_simulation,
            args=(current_params, self.add_status, self.update_progress, self.simulation_complete)
        )
        self.sim_thread.daemon = True
        self.sim_thread.start()

    # Update the plot_results method
    def plot_results(self):
        # 1. Remove the old colorbar if it exists
        if self.colorbar:  # self.colorbar is the Colorbar object from the previous plot
            try:
                self.colorbar.remove()
            except Exception as e:
                self.add_status(f"Log: Issue removing old colorbar (might be okay): {e}\n")
            finally:
                # Essential: Set to None so we don't try to act on a stale/removed object later,
                # and so the 'if self.colorbar:' check is clean next time.
                self.colorbar = None

        # 2. Clear the main axes for the new plot
        self.ax.clear()
        self.ax.set_title("Lid-Driven Cavity Flow")
        
        # Use the global simulation result variables
        global sim_X, sim_Y, sim_p_next, sim_u_next, sim_v_next

        if sim_X is not None and sim_Y is not None and sim_p_next is not None and \
           sim_u_next is not None and sim_v_next is not None:

            # Basic shape check to prevent errors if data is inconsistent
            if sim_X.shape == sim_p_next.shape and sim_X.shape == sim_u_next.shape and sim_X.shape == sim_v_next.shape:
                
                # Check for NaNs or Infs in pressure data which can cause contourf/colorbar issues
                if np.any(np.isnan(sim_p_next)) or np.any(np.isinf(sim_p_next)):
                    self.add_status("Warning: Pressure data contains NaNs or Infs. Plotting may be affected.\n")
                
                try:
                    contour = self.ax.contourf(sim_X[::2, ::2], sim_Y[::2, ::2], sim_p_next[::2, ::2], cmap="jet", levels=20)
                    # 3. Create a new colorbar and store it
                    self.colorbar = self.fig.colorbar(contour, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)
                except Exception as e:
                    self.add_status(f"Error during contourf or colorbar creation: {e}\n")
                    self.ax.text(0.5, 0.5, "Error in plotting data", ha='center', va='center', color='red')
                    self.colorbar = None # Ensure colorbar is None if creation fails

                # Only plot quiver if contour was likely successful (or at least data was valid)
                # and we expect a colorbar or the pressure data wasn't all NaNs
                if contour: # contour will be None if contourf fails badly
                     self.ax.quiver(sim_X[::2, ::2], sim_Y[::2, ::2], sim_u_next[::2, ::2], sim_v_next[::2, ::2], color="black")

            else: # Shape mismatch
                self.add_status("Plotting error: Array shape mismatch.\n")
                self.ax.text(0.5, 0.5, "Error: Data shape mismatch", ha='center', va='center', color='red')
        else: # Data not available
            self.add_status("Plotting error: Simulation data not available to plot.\n")
            self.ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', color='red')

        # Common settings for the axes, applied every time
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        current_domain_size = self.params["DOMAIN_SIZE"].get() # Get current domain size for limits
        self.ax.set_xlim((0, current_domain_size))
        self.ax.set_ylim((0, current_domain_size))
        self.ax.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()

    def stop_simulation(self):
        global simulation_running
        if simulation_running and self.sim_thread and self.sim_thread.is_alive():
            simulation_running = False # Signal the simulation loop to stop
            self.add_status("Attempting to stop simulation...\n")
            self.stop_button.config(state=tk.DISABLED) # Disable while stopping
            # The simulation loop should check `simulation_running` and exit
            # The `completion_callback` will then re-enable buttons.
        else:
            self.add_status("No simulation running or already stopping.\n")


    def _on_closing(self):
        print("Schließe Anwendung...")
        self.shutting_down = True  # Signal für andere Teile der Anwendung

        if hasattr(self, 'stop_simulation'):
            self.stop_simulation()  # Könnte add_status aufrufen

        # Letzte Statusmeldung planen
        if hasattr(self, 'add_status'):
            self.add_status("Anwendung wird geschlossen...\n")

        # Versuchen, alle anstehenden Tkinter-Ereignisse zu verarbeiten
        # Dies gibt den 'after'-Jobs (wie der obigen Statusmeldung) eine Chance, ausgeführt zu werden.
        if hasattr(self.master, 'winfo_exists') and self.master.winfo_exists():
            try:
                self.master.update()  # Verarbeitet alle Events, inkl. Zeichnen
            except tk.TclError:
                # Kann passieren, wenn das Fenster bereits im Zerstörungsprozess ist
                pass 
        
        try:
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
            if hasattr(self, 'fig_analysis') and self.fig_analysis:
                plt.close(self.fig_analysis)
        except Exception as e:
            print(f"Fehler beim Schließen der Matplotlib-Figuren: {e}")
            # Hier kein add_status mehr aufrufen, da wir schon beim Schließen sind
        finally:
            if hasattr(self.master, 'winfo_exists') and self.master.winfo_exists():
                try:
                    self.master.destroy()
                except tk.TclError:
                    # Kann passieren, wenn es bereits zerstört wurde
                    pass

def main_gui():
    plt.style.use("dark_background") 
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
        
    app = NavierStokesGUI(root)
    root.mainloop()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    main_gui()
