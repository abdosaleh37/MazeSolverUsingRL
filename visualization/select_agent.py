import tkinter as tk
from PIL import Image, ImageTk
import ttkbootstrap as tb

def select_agent_window():
    # Create a dictionary to store the selected agent type
    selected_agent = {'type': None}
    
    # Define a function to set the selected agent and close the window
    def set_agent(agent_type):
        selected_agent['type'] = agent_type
        root.destroy()

    # Create the root window
    root = tb.Window(themename="cyborg")  # Use a modern theme
    root.title('Select RL Agent')   # Set the title of the window
    root.geometry("600x400")        # Increased window size
    root.resizable(False, False)
    
    # Load and set background image
    bg_photo = None
    try:
        bg_image = Image.open("assets/background.jpg")
        bg_image = bg_image.resize((600, 400), Image.Resampling.LANCZOS)  # Resize to match new window size
        bg_photo = ImageTk.PhotoImage(bg_image)
        
        # Create a label for the background image
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # If image loads successfully, use a semi-transparent background for UI elements
        title_bg = "#f0f4f8"  # Light blue with some transparency
        frame_bg = "#f0f4f8"
    except Exception as e:
        print(f"Could not load background image: {e}")
        # Use default background color for UI elements
        root.configure(bg="#1a1a1a")
        title_bg = "#f0f4f8"
        frame_bg = "#f0f4f8"

    # Canvas for semi-transparent title background
    canvas = tk.Canvas(root, width=600, height=80, highlightthickness=0)
    canvas.place(relx=0.5, rely=0.68, anchor='center')
    # Draw a semi-transparent rounded rectangle
    canvas.create_rectangle(80, 10, 520, 70, fill="#222222", outline="#222222", width=0)
    # Title text
    canvas.create_text(300, 40, text="Choose an agent to train:", font=("Segoe UI", 22, "bold"),
                       fill="#fff", anchor='center')

    # Use ttkbootstrap's rounded buttons
    style = tb.Style()
    style.configure("Custom.TButton",
                    font=("Segoe UI", 16, "bold"),
                    padding=10)

    # Create a frame to hold both buttons, centered at the bottom
    btn_frame = tk.Frame(root, bg="#8f5cff", highlightthickness=0)
    btn_frame.place(relx=0.5, rely=0.88, anchor='center')

    q_btn = tb.Button(
        btn_frame, text='Q-Learning', bootstyle="info-outline", style="Custom.TButton",
        width=18,  # Slightly reduced width
        command=lambda: set_agent('q')
    )
    q_btn.pack(side='left', padx=(0, 15), ipady=10)  # 15px space to the right

    pg_btn = tb.Button(
        btn_frame, text='Policy Gradient', bootstyle="info-outline", style="Custom.TButton",
        width=18,  # Slightly reduced width
        command=lambda: set_agent('pg')
    )
    pg_btn.pack(side='left', padx=(15, 0), ipady=10)  # 15px space to the left

    # Center the window on the screen
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
    
    # Return the selected agent type after the window is closed
    return selected_agent
