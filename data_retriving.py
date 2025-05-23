import time
import re
from data_api import chat_with_mistral
import pandas as pd
from tqdm import tqdm


prompts = {"positive" : "As an actor would do, impersonate a 25-year-old man delivering a monologue to himself, as if no one were listening — a spontaneous stream of thought. He works as a data scientist in a large company in London. He feels genuinely happy with the choice he made and fulfilled by his new job. Having recently moved to the city, he is still adapting to this new chapter of his life, but he loves the energy of London and embraces its fast-paced lifestyle. He spends most of his time either working or enjoying the company of his amazing friends. He’s also starting to explore the city on his own and reflect on new personal and professional goals for the future.",
           "negative" : "As an actor would do, impersonate a 25-year-old man delivering a monologue to himself, as if no one were listening — a spontaneous stream of thought. He works as a data scientist in a large company in London. He feels lost and  hates the city. He fells disconnected from the people around him and overwhelmed by the fast-paced environment. He often questions his abilities and his thinking about that accepting the job was not the right decision at all. Although surrounded by colleagues and crowds, a deep sense of loneliness and doubt weighs on him as he struggles to find meaning and stability in this new phase of his life."
           }
datas = []

# Dataset feature
count_text = list(range(0,100))
temperature_level = [0.1,0.7,1.3]

# iterations count
total_iterations = len(prompts) * len(count_text) * len(temperature_level)

with tqdm(total=total_iterations, desc="Generating letters") as pbar:
    for prompt_type, prompt_text in prompts.items():
        for i in temperature_level:
            for _ in count_text:
                chat_history = [
                    {"role": "user", "content": prompt_text}
                ]

                try:
                    # Send the message and receive the model's response
                    response = chat_with_mistral(chat_history, i)
                    response_clean = re.sub(r"\[.*?\]|\(.*?\)", "", response)
                    response_clean = response_clean.replace("\n", " ").replace("\t", " ").replace("\r", " ")
                    datas.append({"type of prompt": prompt_type,
                        "temperature": i,
                                  "text": response_clean
                                  })
                    print("Assistant:", response_clean)
                except Exception as e:
                    print("Error:", str(e))
                time.sleep(3)
                pbar.update(1)

dataspd = pd.DataFrame(datas)
dataspd.to_csv("data_chat_100.csv")
