import tkinter as tk

def select_agent_window():
    selected_agent = {'type': None}
    def set_agent(agent_type):
        selected_agent['type'] = agent_type
        root.destroy()

    root = tk.Tk()
    root.title('Select RL Agent')
    root.geometry("500x250")
    root.configure(bg="#f0f4f8")

    # Title label
    title = tk.Label(root, text='Choose an agent to train:', font=("Segoe UI", 18, "bold"), bg="#f0f4f8", fg="#333")
    title.pack(pady=(30, 20))

    # Button frame for centering
    btn_frame = tk.Frame(root, bg="#f0f4f8")
    btn_frame.pack(pady=10)

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

    q_btn = tk.Button(btn_frame, text='Q-Learning', command=lambda: set_agent('q'), **btn_style)
    q_btn.grid(row=0, column=0, padx=20, pady=10)

    pg_btn = tk.Button(btn_frame, text='Policy Gradient', command=lambda: set_agent('pg'), **btn_style)
    pg_btn.grid(row=0, column=1, padx=20, pady=10)

    # Optional: Add a subtle border or shadow effect
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
    
    return selected_agent
