import json
import random
import os

TEMPLATES = [
    {
        "domain": "hotel_booking",
        "scenarios": [
            {"inp": "Enikku kochi yil oru hotel room book cheyyanam.", "out": "Sure, Kochi-yil ethra divasatheykkanu room vendathu? (How many days do you need the room for in Kochi?)"},
            {"inp": "Munnar il family suite lathyam aano?", "out": "Njangal Munnar-il family suites check cheyyam. Athinte price details veno? (Should I check the price details for Munnar family suites?)"},
            {"inp": "Hotel il AC room undo?", "out": "Athe, AC rooms available aanu. Ethra perundu ningalude koode? (Yes, AC rooms are available. How many people are with you?)"}
        ]
    },
    {
        "domain": "medical_appointment",
        "scenarios": [
            {"inp": "Nale oru ENT doctor-e kananam.", "out": "Nale morning slots available aanu. Ningalude convenient time parayam? (What is your convenient time for tomorrow morning?)"},
            {"inp": "Cardiologist de appointment kittumo?", "out": "Dr. Joseph available aanu. Ningalude contact number tharumo booking-nu? (Dr. Joseph is available. Can you provide your contact number for booking?)"}
        ]
    },
    {
        "domain": "academic_consultation",
        "scenarios": [
            {"inp": "Abroad study-ne patti consultation venam.", "out": "Athe, njangalude experts help cheyyam. Ningalude interest ulla country ethannu? (Yes, our experts can help. Which country are you interested in?)"},
            {"inp": "MBA course fees detail venam.", "out": "MBA-de detailed brochure njan send cheyyam. Ningalude email tharumo? (I will send the MBA brochure. Can you provide your email?)"}
        ]
    }
]

MALAYALAM_VOCAB = {
    "hotel": "ഹോട്ടൽ", "room": "റൂം", "book": "ബുക്ക്", "kochi": "കൊച്ചി", 
    "doctor": "ഡോക്ടർ", "appointment": "അപ്പോയിന്റ്മെന്റ്", "fees": "ഫീസ്"
}

def format_qwen_chatml(system, user, assistant):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

def generate(output_path, n=500):
    system = "You are StrawCore AI, a client management assistant speaking Manglish and Malayalam script fluently."
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            temp = random.choice(TEMPLATES)
            scenario = random.choice(temp["scenarios"])
            
            # 50% chance to convert input to native Malayalam script
            user_msg = scenario["inp"]
            if random.random() > 0.5:
                for k, v in MALAYALAM_VOCAB.items():
                    user_msg = user_msg.replace(k, v)
                    
            text = format_qwen_chatml(system, user_msg, scenario["out"])
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Generated {n} samples in {output_path}")

if __name__ == "__main__":
    generate("data/manglish_booking_data.jsonl", 500)
