from env import MazeEnv
import time

if __name__ == "__main__":
    env = MazeEnv()  # إنشاء البيئة
    state = env.reset()  # إعادة البيئة لحالتها الأولية
    done = False
    
    while not done:
        action = env.action_space.sample()  # اختيار حركة عشوائية
        state, reward, done, _ = env.step(action)  # تنفيذ الحركة
        env.render()  # عرض المتاهة بعد الحركة
        
        # طباعة معلومات عن الحالة والمكافأة
        print(f"Agent Position: {state}, Reward: {reward}")
        
        # إضافة تأخير بسيط بين الحركات لرؤية التغييرات بوضوح
        time.sleep(0.1)  # تأخير نصف ثانية بين كل خطوة وأخرى

    print("Reached the goal!")  # عند الوصول للنهاية
