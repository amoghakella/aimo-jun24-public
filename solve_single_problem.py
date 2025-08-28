# original notebook: https://www.kaggle.com/code/abdurrafae/improved-code-interpretation

import time
from tqdm import tqdm
import torch
import gc
torch.backends.cuda.enable_mem_efficient_sdp(False)
import transformers
from transformers import (
    set_seed,
    StoppingCriteriaList
)
print(f"Transformers Version: {transformers.__version__}")
set_seed(42)

import numpy as np
from numpy.random import choice

from utils import process_text_output, summarize_results, accumulate_prompt_code, get_model_response, StoppingCriteriaSub, train_subj, test_subj
from configs import *

from pdb import set_trace as bp


torch.cuda.empty_cache()
gc.collect()


class AIMO_Solver(object):
    def __init__(self, model, tokenizer, pipeline):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        if isinstance(repetition, int) and repetition > 0:
            self.repetition = repetition
        else:
            self.repetition = N_REPETITIONS

        self.stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in STOP_WORDS]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])

        code = """Below is a math problem you are to solve (positive numerical answer):
        \"{}\"
        To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
        Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.

        Approach:"""

        cot = """Below is a math problem you are to solve (positive numerical answer!):
        \"{}\"
        Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{}.\n\n"""

        tool_instruction = '\n\nPlease integrate natural language reasoning with programs to solve the above problem, and put your final numerical answer within \\boxed{}.\nNote that the intermediary calculations may be real numbers, but the final numercal answer would always be an integer.'
        # tool_instruction = " The answer should be given as a non-negative modulo 1000."
        # tool_instruction += '\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.'

        self.promplt_options = [code, cot]

        self.i_problem = 0

        self.total_results = {} # problem_id: list of final results extracted by process_text_output
        self.total_answers = {} # problem_id: list of code output
        self.total_outputs = {} # problem_id: list of combine output from code & text
        self.best_stats = {} # problem_id: the most common from Counter(outputs).most_common()
        self.question_type_counts = {} # problem_id: list of (valid_text, valid_code)
        train_subj()
        
        super().__init__()

    def predict(self, problem):
        if isinstance(repetition, int) and repetition > 0:
            self.repetition = repetition
        subj = test_subj(problem)
        
        for i_repeat in tqdm(range(N_REPETITIONS)):

            print(f"\n\n\n########## QUESTION {self.i_problem} - {i_repeat}")

            # if best_stats.get(self.i_problem, (-1, -1))[1] > np.sqrt(i_repeat):
            if self.best_stats.get(self.i_problem, (-1, -1))[1] > N_REPETITIONS/2 or self.best_stats.get(self.i_problem, (-1, -1))[1] > CUTOFF_OCCURANCE_BEST:
                print("########## SKIPPING CAUSE ALREADY FOUND BEST")
                continue

            text_answers, code_answers = self.question_type_counts.get(self.i_problem, PROMPT_COUNT_INIT)

            for _ in range(5):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.2)

            try:
                ALREADY_GEN = 0
                code_error = None
                code_error_count = 0
                code_output = -1
                # initail_message = problem + tool_instruction 

                counts = np.array([text_answers, code_answers])
                

                    initail_message = draw[0].format(problem, "{}") # insert problem into prompt
                    prompt = f"User: {initail_message}"

                    current_printed = len(prompt) # keep track of length (end) of model response
                    # print(f"{i_repeat}_{prompt}\n")
                    print(f">>>>>>>>>> PROMPT\n{prompt}\n")

                    model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                    input_len = len(model_inputs['input_ids'][0])
                    model_inputs, ALREADY_GEN, generation_output, output_ids, decoded_output, current_printed = get_model_response(prompt, self.model, self.tokenizer, None, current_printed, 0, TEMPERATURE, TOP_P, self.stopping_criteria, USE_PAST_KEY, TOTAL_TOKENS)

                    cummulative_code = ""

                    stop_word_cond = False # TODO force go into while loop
                    stop_word_cond = False
                    for stop_word in STOP_WORDS:
                        stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):] == stop_word)
                    if not stop_word_cond:
                        print('!!!!!!!!!! NO STOP_WORD FOUND.')

                    while (stop_word_cond) and (ALREADY_GEN<TOTAL_TOKENS):


                        prompt, code_output, cummulative_code, temperature_inner, top_p_inner, code_error, code_error_count, break_for_error = accumulate_prompt_code(decoded_output, code_output, cummulative_code, code_error, code_error_count, TEMPERATURE_CODING, TOP_P_CODING, TEMPERATURE, TOP_P, timeout=timeout)
                        if break_for_error:
                            break

                        model_inputs, ALREADY_GEN, generation_output, output_ids, decoded_output, current_printed = get_model_response(prompt, self.model, self.tokenizer, generation_output, current_printed, input_len, temperature_inner, top_p_inner, self.stopping_criteria, USE_PAST_KEY, TOTAL_TOKENS)

                        stop_word_cond = False
                        for stop_word in STOP_WORDS:
                            stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):] == stop_word)
                        if not stop_word_cond:
                            print('!!!!!!!!!! NO STOP_WORD FOUND.')

                    print('########## ALREADY_GEN:', ALREADY_GEN, '; TOTAL_TOKENS:', TOTAL_TOKENS)
                    if ALREADY_GEN >= TOTAL_TOKENS:
                        print('!!!!!!!!!! ALREADY_GEN >= TOTAL_TOKENS')

                    if USE_PAST_KEY:
                        output_ids = generation_output.sequences[0]
                    else:
                        output_ids = generation_output[0]

                    raw_output = self.tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
                    # print(f"\n\nOutput :\n{raw_output}\n")
                    result_output = process_text_output(raw_output)

                    try:
                        code_output = round(float(eval(code_output))) % 1000 # TODO too ad hoc...
                    except Exception as e:
                        print(e, 'final_eval')
                        code_output = -1

            except Exception as e:
                print(e, "5")
                result_output, code_output = -1, -1

            # summarize results
            _occurance_best = summarize_results(self.i_problem, i_repeat, code_output, result_output, code_answers, text_answers, self.total_outputs, self.total_results, self.total_answers, self.best_stats, self.question_type_counts)

            print("######### code_answers", code_answers - PROMPT_COUNT_INIT[1], "text_answers", text_answers - PROMPT_COUNT_INIT[0])
            if DEBUG:
                break

        prediction = self.best_stats[self.i_problem][0]
        self.i_problem += 1
        return prediction
