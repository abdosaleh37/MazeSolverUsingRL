import tkinter as tk
from PIL import Image, ImageTk

def select_agent_window():
    # Create a dictionary to store the selected agent type
    selected_agent = {'type': None}
    
    # Define a function to set the selected agent and close the window
    def set_agent(agent_type):
        selected_agent['type'] = agent_type
        root.destroy()

    # Create the root window
    root = tk.Tk()
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
        root.configure(bg="#f0f4f8")
        title_bg = "#f0f4f8"
        frame_bg = "#f0f4f8"

    # Create a frame for title and buttons at the bottom
    bottom_frame = tk.Frame(root, highlightthickness=0)
    bottom_frame.place(relx=0.5, rely=1.0, anchor='s', relwidth=1.0, y=-40)  # 40px above the bottom

    # Create a label that will display the title on the window
    title = tk.Label(bottom_frame, text='Choose an agent to train:', font=("Segoe UI", 20, "bold"), fg="#333")
    title.pack(pady=(0, 20))   # Reduced top padding, increased bottom padding

    # Create a frame to hold the buttons for selecting the agent
    btn_frame = tk.Frame(bottom_frame)
    btn_frame.pack()

    # Define a style for the buttons
    btn_style = {
        "width": 20,  # Increased button width
        "height": 2,
        "font": ("Segoe UI", 14, "bold"),
        "bg": "#4f8cff",
        "fg": "white",
        "activebackground": "#357ae8",
        "activeforeground": "white",
        "bd": 0,
        "relief": "flat",
        "cursor": "hand2"
    }

    # Create the "Q-Learning" button and assign the corresponding action
    q_btn = tk.Button(btn_frame, text='Q-Learning', command=lambda: set_agent('q'), **btn_style)
    q_btn.grid(row=0, column=0, padx=25, pady=10)  # Increased padding between buttons

    # Create the "Policy Gradient" button and assign the corresponding action
    pg_btn = tk.Button(btn_frame, text='Policy Gradient', command=lambda: set_agent('pg'), **btn_style)
    pg_btn.grid(row=0, column=1, padx=25, pady=10)  # Increased padding between buttons

    # Center the window on the screen
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
    
    # Return the selected agent type after the window is closed
    return selected_agent
