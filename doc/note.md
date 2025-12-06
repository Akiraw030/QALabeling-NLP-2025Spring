# This is some implemetation note

## The grouping of output features

### Six head, QA tangled:

[Group 1]:
- question_multi_intent
- question_type_choice
- question_type_reason_explanation
- answer_type_reason_explanation

[Group 2]:
- question_asker_intent_understanding
- question_body_critical
- question_interestingness_others
- question_interestingness_self
- question_well_written

[Group 3]:
- question_conversational
- question_not_really_a_question
- question_opinion_seeking
- question_type_compare
- question_type_consequence
- question_type_definition
- question_type_entity
- question_type_spelling

[Group 4]:
- answer_helpful
- answer_level_of_information
- answer_plausible
- answer_relevance
- answer_satisfaction
- answer_well_written

[Group 5]:
- question_fact_seeking
- question_type_procedure
- answer_type_procedure

[Group 6]:
- question_expect_short_answer
- question_has_commonly_accepted_answer
- question_type_instructions
- answer_type_instructions

### Six head, QA splited:
[Group 1]:
- question_expect_short_answer
- question_fact_seeking
- question_has_commonly_accepted_answer
- question_type_instructions
- question_type_procedure

[Group 2]:
- question_asker_intent_understanding
- question_body_critical
- question_interestingness_others
- question_interestingness_self
- question_well_written

[Group 3]:
- question_conversational
- question_opinion_seeking

[Group 4]:
- question_multi_intent
- question_not_really_a_question
- question_type_choice
- question_type_compare
- question_type_consequence
- question_type_definition
- question_type_entity
- question_type_reason_explanation
- question_type_spelling

[Group 1]:
- answer_type_instructions
- answer_type_procedure

[Group 2]:
- answer_helpful
- answer_level_of_information
- answer_plausible
- answer_relevance
- answer_satisfaction
- answer_type_reason_explanation
- answer_well_written

### The performance of different projector head grouping strategy - With 6 head QA distangled later

Below experiment is done under the setting of one Deberta and Kflod

|Stragegy|Fold0Epoch2(Train)|Fold1Epoch2(Train)|Fold2Epoch2(Train)|Fold3Epoch2(Train)|Fold4Epoch2(Train)|Public(Eval)|Private(Eval)|
|-|-|-|-|-|-|-|-|
|One global head||||||0.30509|0.28049|
|Two head, QA splited|Loss: 0.3770 - Raw Score: 0.3391|Loss: 0.3797 - Raw Score: 0.3287|Loss: 0.3779 - Raw Score: 0.3360|Loss: 0.3793 - Raw Score: 0.3207|Loss: 0.3788 - Raw Score: 0.3283|0.31465|0.29607|
|Six head, global|Loss: 0.3750 - Raw Score: 0.3413|Loss: 0.3775 - Raw Score: 0.3363|Loss: 0.3767 - Raw Score: 0.3392|Loss: 0.3783 - Raw Score: 0.3239|Loss: 0.3733 - Raw Score: 0.3449|0.31958|0.30399|
|Six head, QA splited|Loss: 0.3730 - Raw Score: 0.3525|Loss: 0.3776 - Raw Score: 0.3354|Loss: 0.3767 - Raw Score: 0.3333|Loss: 0.3788 - Raw Score: 0.3285|Loss: 0.3784 - Raw Score: 0.3295|0.31968|0.30270|

## The use of class weight CE loss - With class weight later

Below experiment is done under the setting of one Deberta and six head, QA splited

|Stragegy|Fold0Epoch2(Train)|Fold1Epoch2(Train)|Fold2Epoch2(Train)|Fold3Epoch2(Train)|Fold4Epoch2(Train)|Public(Eval)|Private(Eval)|
|-|-|-|-|-|-|-|-|
|Without weight|Loss: 0.3730 - Raw Score: 0.3525|Loss: 0.3776 - Raw Score: 0.3354|Loss: 0.3767 - Raw Score: 0.3333|Loss: 0.3788 - Raw Score: 0.3285|Loss: 0.3784 - Raw Score: 0.3295|0.31968|0.30270|
|With weight|Loss: 0.3964 - Raw Score: 0.3555|Loss: 0.4015 - Raw Score: 0.3331|Loss: 0.3990 - Raw Score: 0.3412|Epoch 2 - Loss: 0.4013 - Raw Score: 0.3379|Epoch 2 - Loss: 0.3999 - Raw Score: 0.3431|0.32706|0.31128|

## The use of larger lr for head - With larger lr for head later

|Stragegy|Fold0Epoch2(Train)|Fold1Epoch2(Train)|Fold2Epoch2(Train)|Fold3Epoch2(Train)|Fold4Epoch2(Train)|Public(Eval)|Private(Eval)|
|-|-|-|-|-|-|-|-|
|Smaller LR|Loss: 0.3730 - Raw Score: 0.3525|Loss: 0.3776 - Raw Score: 0.3354|Loss: 0.3767 - Raw Score: 0.3333|Loss: 0.3788 - Raw Score: 0.3285|Loss: 0.3784 - Raw Score: 0.3295|0.31968|0.30270|
|Larger LR|Loss: 0.3894 - Raw Score: 0.3876|Loss: 0.3905 - Raw Score: 0.3744|Loss: 0.3913 - Raw Score: 0.3759|Loss: 0.3911 - Raw Score: 0.3653|Epoch 2 - Loss: 0.3906 - Raw Score: 0.3718|0.36439|0.34443|

## The use of larger epoch - With 5 epochs later

|Stragegy|Public(Eval)|Private(Eval)|
|-|-|-|
|2 epoch|0.36439|0.34443|
|5 epoch|0.39181|0.37820|

## The use of group k fold - Without group k fold

|Stragegy|Public(Eval)|Private(Eval)|
|-|-|-|
|5 fold|0.39181|0.37820|
|no fold|0.38451|0.37011|

## The strategy of head output

https://manikanthgoud123.medium.com/google-quest-q-a-labeling-kaggle-competition-d205bea1e026
question_asker_inten: no. of unique label values: 9
question_body_critic: no. of unique label values: 9
question_conversatio: no. of unique label values: 5
question_expect_shor: no. of unique label values: 5
question_fact_seekin: no. of unique label values: 5
question_has_commonl: no. of unique label values: 5
question_interesting: no. of unique label values: 9
question_interesting: no. of unique label values: 9
question_multi_inten: no. of unique label values: 5
question_not_really_: no. of unique label values: 5
question_opinion_see: no. of unique label values: 5
question_type_choice: no. of unique label values: 5
question_type_compar: no. of unique label values: 5
question_type_conseq: no. of unique label values: 5
question_type_defini: no. of unique label values: 5
question_type_entity: no. of unique label values: 5
question_type_instru: no. of unique label values: 5
question_type_proced: no. of unique label values: 5
question_type_reason: no. of unique label values: 5
question_type_spelli: no. of unique label values: 3
question_well_writte: no. of unique label values: 9
answer_helpful: no. of unique label values: 9
answer_level_of_info: no. of unique label values: 9
answer_plausible: no. of unique label values: 9
answer_relevance: no. of unique label values: 9
answer_satisfaction: no. of unique label values: 17
answer_type_instruct: no. of unique label values: 5
answer_type_procedur: no. of unique label values: 5
answer_type_reason_e: no. of unique label values: 5
answer_well_written: no. of unique label values: 9

|Stragegy|Public(Eval)|Private(Eval)|
|-|-|-|
|Single regression|||
|Ordinal regression|||

## The post processing rounding trick

|Stragegy|Public(Eval)|Private(Eval)|
|-|-|-|
|3rd placed OptimizedRounder|||
|4th placed Rounding trick|||
|1st placed distribution|||

## The use of differnet model

|Model|Public(Eval)|Private(Eval)|
|-|-|-|
|deberta-v3-base|||
|Llama|||
|Qwen|||
|roberta|||
|Mistral|||
|Phi|||
|bge|||
|instructor|||
|E5|||