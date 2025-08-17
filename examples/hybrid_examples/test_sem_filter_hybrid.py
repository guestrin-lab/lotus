import logging
import os
import pandas as pd
import sys
import time
from pathlib import Path
from typing import List, Tuple

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import lotus
from lotus.models import LM 
import lotus.sem_ops.sem_filter_hybrid

def calculate_precision_recall(result: pd.DataFrame, expected: List[bool]) -> Tuple[float, float]:
    if 'sentiment' in result.columns:
        actual = [True if val == 'positive' else False for val in result['sentiment']]
    else:
        actual = [idx in result.index for idx in range(len(expected))]
    
    true_positives = sum(1 for a, e in zip(actual, expected) if a and e)
    false_positives = sum(1 for a, e in zip(actual, expected) if a and not e)
    false_negatives = sum(1 for a, e in zip(actual, expected) if not a and e)

    precision = (
        true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    )
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return precision, recall

def run_comparison_test(df: pd.DataFrame, user_instruction: str, expected_condition: pd.Series, model):

    all_model_results = {}
    
    for model_name in models_to_test:
        lm = LM(model=model_name)
        lotus.settings.configure(lm=lm)
            
        results_store = {} 
        try:
            print("sem_filter (baseline)")
            lm.reset_stats()
            start_time = time.time()
            result_standard = df.sem_filter(user_instruction, return_explanations=True)
            standard_time = time.time() - start_time

            std_precision, std_recall = calculate_precision_recall(result_standard, expected_condition.tolist())
            std_f1 = 2 * (std_precision * std_recall) / (std_precision + std_recall) if (std_precision + std_recall) > 0 else 0

            results_store["sem_filter"] = {
                "Time": standard_time,
                "Precision": std_precision,
                "Recall": std_recall,
                "F1": std_f1,
            }
            lm.reset_stats()

            # define different accuracy vs. cost trade-off levels
            preference_levels = [0.0, 0.5, 0.8, 0.9] 
            default_kw_sample_perc = 0.1 
            default_num_kw_calls = 1

            print("sem_filter_hybrid (hybrid)")
            for pref in preference_levels:
                if pref < 0.83:
                    run_label = f"Hybrid (Pref={pref:.1f}, KW=Exact)"
                    lm.reset_stats()
                    start_time = time.time()
                    
                    df_hybrid = df.sem_filter_hybrid(
                        user_instruction,
                        return_explanations=True,
                        accuracy_cost_preference=pref,
                        num_keyword_calls=default_num_kw_calls,
                        sample_percentage_for_keywords=default_kw_sample_perc,
                    )
                    
                    end_time = time.time()
                    hybrid_time = end_time - start_time
                    hybrid_precision, hybrid_recall = calculate_precision_recall(df_hybrid, expected_condition.tolist())
                    hybrid_f1 = 2 * (hybrid_precision * hybrid_recall) / (hybrid_precision + hybrid_recall) if (hybrid_precision + hybrid_recall) > 0 else 0
                    results_store[run_label] = {
                        "Time": hybrid_time,
                        "Precision": hybrid_precision,
                        "Recall": hybrid_recall,
                        "F1": hybrid_f1,
                    }

                else:
                    run_label = f"Hybrid (Pref={pref:.1f}, Fallback)"
                    lm.reset_stats()
                    start_time = time.time()
                    df_hybrid = df.sem_filter_hybrid(
                        user_instruction,
                        return_explanations=True,
                        accuracy_cost_preference=pref,
                        num_keyword_calls=default_num_kw_calls,
                        sample_percentage_for_keywords=default_kw_sample_perc,
                    )
                    end_time = time.time()
                    hybrid_time = end_time - start_time
                    hybrid_precision, hybrid_recall = calculate_precision_recall(df_hybrid, expected_condition.tolist())
                    hybrid_f1 = 2 * (hybrid_precision * hybrid_recall) / (hybrid_precision + hybrid_recall) if (hybrid_precision + hybrid_recall) > 0 else 0
                    results_store[run_label] = {
                        "Time": hybrid_time, 
                        "Precision": hybrid_precision,
                        "Recall": hybrid_recall,
                        "F1": hybrid_f1,
                    }
        
        except Exception as model_test_e:
             logging.error(f"Error running tests for model {model_name}: {model_test_e}", exc_info=True)
        
        all_model_results[model_name] = results_store 

    display_data = []
    for model, results in all_model_results.items():
        for method, metrics in results.items():
             metrics["Model"] = model
             metrics["Method"] = method
             display_data.append(metrics)
    
    results_df = pd.DataFrame(display_data)
    cols_to_show = ["Model", "Method", "Time", "F1", "Precision", "Recall"]
    print(results_df[cols_to_show].to_string())


if __name__ == "__main__":
    # To run the test on the reviews.csv dataset, uncomment the following lines
    # Put the reviews.csv file in the data folder (../data/reviews.csv)

    # data_path = os.path.join(project_root, "data", "reviews.csv")
    # full_df = pd.read_csv(data_path)


    # make dataset similar to the reviews.csv dataset
    # Some runs on this sample dataset:
    #                  Model                       Method       Time        F1  Precision  Recall
    # 0  ollama/llama3.1                   sem_filter  21.813436  0.545455   0.500000     0.6
    # 1  ollama/llama3.1  Hybrid (Pref=0.0, KW=Exact)   7.121476  0.500000   0.500000     0.5
    # 2  ollama/llama3.1  Hybrid (Pref=0.5, KW=Exact)   3.381731  0.363636   0.333333     0.4
    # 3  ollama/llama3.1  Hybrid (Pref=0.8, KW=Exact)   5.168907  0.000000   0.000000     0.0
    # 4  ollama/llama3.1  Hybrid (Pref=0.9, Fallback)  18.838958  0.545455   0.500000     0.6
    full_df = pd.DataFrame({
        "review": [
            "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.",
            "A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only ""has got all the polari"" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great master's of comedy and his life. <br /><br />The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional 'dream' techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell's murals decorating every surface) are terribly well done.",
            "I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching a light-hearted comedy. The plot is simplistic, but the dialogue is witty and the characters are likable (even the well bread suspected serial killer). While some may be disappointed when they realize this is not Match Point 2: Risk Addiction, I thought it was proof that Woody Allen is still fully in control of the style many of us have grown to love.<br /><br />This was the most I'd laughed at one of Woody's comedies in years (dare I say a decade?). While I've never been impressed with Scarlet Johanson, in this she managed to tone down her ""sexy"" image and jumped right into a average, but spirited young woman.<br /><br />This may not be the crown jewel of his career, but it was wittier than ""Devil Wears Prada"" and more interesting than ""Superman"" a great comedy to go see with friends.",
            "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.",
            "Petter Mattei's ""Love in the Time of Money"" is a visually stunning film to watch. Mr. Mattei offers us a vivid portrait about human relations. This is a movie that seems to be telling us what money, power and success do to people in the different situations we encounter. <br /><br />This being a variation on the Arthur Schnitzler's play about the same theme, the director transfers the action to the present time New York where all these different characters meet and connect. Each one is connected in one way, or another to the next person, but no one seems to know the previous point of contact. Stylishly, the film has a sophisticated luxurious look. We are taken to see how these people live and the world they live in their own habitat.<br /><br />The only thing one gets out of all these souls in the picture is the different stages of loneliness each one inhabits. A big city is not exactly the best place in which human relations find sincere fulfillment, as one discerns is the case with most of the people we encounter.<br /><br />The acting is good under Mr. Mattei's direction. Steve Buscemi, Rosario Dawson, Carol Kane, Michael Imperioli, Adrian Grenier, and the rest of the talented cast, make these characters come alive.<br /><br />We wish Mr. Mattei good luck and await anxiously for his next work.",
            "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring. It just never gets old, despite my having seen it some 15 or more times in the last 25 years. Paul Lukas' performance brings tears to my eyes, and Bette Davis, in one of her very few truly sympathetic roles, is a delight. The kids are, as grandma says, more like ""dressed-up midgets"" than children, but that only makes them more fun to watch. And the mother's slow awakening to what's happening in the world and under her own roof is believable and startling. If I had a dozen thumbs, they'd all be ""up"" for this movie.",
            "I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today it would bring back the kid excitement in me.I grew up on black and white TV and Seahunt with Gunsmoke were my hero's every week.You have my vote for a comeback of a new sea hunt.We need a change of pace in TV and this would work for a world of under water adventure.Oh by the way thank you for an outlet like this to view many viewpoints about TV and the many movies.So any ole way I believe I've got what I wanna say.Would be nice to read some more plus points about sea hunt.If my rhymes would be 10 lines would you let me submit,or leave me out to be in doubt and have me to quit,If this is so then I must go so lets do it.",
            "This show was an amazing, fresh & innovative idea in the 70's when it first aired. The first 7 or 8 years were brilliant, but things dropped off after that. By 1990, the show was not really funny anymore, and it's continued its decline further to the complete waste of time it is today.<br /><br />It's truly disgraceful how far this show has fallen. The writing is painfully bad, the performances are almost as bad - if not for the mildly entertaining respite of the guest-hosts, this show probably wouldn't still be on the air. I find it so hard to believe that the same creator that hand-selected the original cast also chose the band of hacks that followed. How can one recognize such brilliance and then see fit to replace it with such mediocrity? I felt I must give 2 stars out of respect for the original cast that made this show such a huge success. As it is now, the show is just awful. I can't believe it's still on the air.",
            "Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake. I've seen 950+ films and this is truly one of the worst of them - it's awful in almost every way: editing, pacing, storyline, 'acting,' soundtrack (the film's only song - a lame country tune - is played no less than four times). The film looks cheap and nasty and is boring in the extreme. Rarely have I been so happy to see the end credits of a film. <br /><br />The only thing that prevents me giving this a 1-score is Harvey Keitel - while this is far from his best performance he at least seems to be making a bit of an effort. One for Keitel obsessives only.",
            "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.<br /><br />Great Camp!!!",
            "Phil the Alien is one of those quirky films where the humour is based around the oddness of everything rather than actual punchlines.<br /><br />At first it was very odd and pretty funny but as the movie progressed I didn't find the jokes or oddness funny anymore.<br /><br />Its a low budget film (thats never a problem in itself), there were some pretty interesting characters, but eventually I just lost interest.<br /><br />I imagine this film would appeal to a stoner who is currently partaking.<br /><br />For something similar but better try ""Brother from another planet""",
            "I saw this movie when I was about 12 when it came out. I recall the scariest scene was the big bird eating men dangling helplessly from parachutes right out of the air. The horror. The horror.<br /><br />As a young kid going to these cheesy B films on Saturday afternoons, I still was tired of the formula for these monster type movies that usually included the hero, a beautiful woman who might be the daughter of a professor and a happy resolution when the monster died in the end. I didn't care much for the romantic angle as a 12 year old and the predictable plots. I love them now for the unintentional humor.<br /><br />But, about a year or so later, I saw Psycho when it came out and I loved that the star, Janet Leigh, was bumped off early in the film. I sat up and took notice at that point. Since screenwriters are making up the story, make it up to be as scary as possible and not from a well-worn formula. There are no rules.",
        ],
        "sentiment": [ "positive", "positive", "positive", "negative", "positive", "negative",
                      "positive", "negative", "negative", "positive", "negative", "negative", ]
    })
    
    sample_size = len(full_df)
    df = full_df.sample(n=sample_size, random_state=42)

    models_to_test = ["ollama/llama3.1"]
    # models_to_test = ["gpt-4o-mini"]
    user_instruction = "Extract positive reviews from {review}"
    expected_condition = df['sentiment'] == "positive"

    run_comparison_test(
        df,
        user_instruction=user_instruction, 
        expected_condition=expected_condition, 
        model=models_to_test, 
    ) 

