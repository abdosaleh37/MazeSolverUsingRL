import tkinter as tk

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
    root.geometry("500x250")        # Set the window size
    root.configure(bg="#f0f4f8")    # Set the background color

    # Create a label that will display the title on the window
    title = tk.Label(root, text='Choose an agent to train:', font=("Segoe UI", 18, "bold"), bg="#f0f4f8", fg="#333")
    title.pack(pady=(30, 20))   # Add the label to the window with padding

    # Create a frame to hold the buttons for selecting the agent
    btn_frame = tk.Frame(root, bg="#f0f4f8")
    btn_frame.pack(pady=10)

    # Define a style for the buttons
    btn_style = {
        "width": 18,
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
    q_btn.grid(row=0, column=0, padx=20, pady=10)

    # Create the "Policy Gradient" button and assign the corresponding action
    pg_btn = tk.Button(btn_frame, text='Policy Gradient', command=lambda: set_agent('pg'), **btn_style)
    pg_btn.grid(row=0, column=1, padx=20, pady=10)

    # Center the window on the screen
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
    
    # Return the selected agent type after the window is closed
    return selected_agent
