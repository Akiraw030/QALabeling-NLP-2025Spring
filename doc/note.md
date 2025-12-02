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

### The performance of different projector head grouping strategy

Below experiment is done under the setting of one Deberta and Kflod

|Stragegy|Fold0Epoch2(Train)|Fold1Epoch2(Train)|Fold2Epoch2(Train)|Fold3Epoch2(Train)|Fold4Epoch2(Train)|Public(Eval)|Private(Eval)|
|-|-|-|-|-|-|-|-|
|One global head||||||0.30509|0.28049|
|Two head, QA splited|Loss: 0.3770 - Raw Score: 0.3391|Loss: 0.3797 - Raw Score: 0.3287|Loss: 0.3779 - Raw Score: 0.3360|Loss: 0.3793 - Raw Score: 0.3207|Loss: 0.3788 - Raw Score: 0.3283|0.31465|0.29607|
|Six head, global|Loss: 0.3750 - Raw Score: 0.3413|Loss: 0.3775 - Raw Score: 0.3363|Loss: 0.3767 - Raw Score: 0.3392|Loss: 0.3783 - Raw Score: 0.3239|Loss: 0.3733 - Raw Score: 0.3449|0.31958|0.30399|
|Six head, QA splited|Loss: 0.3730 - Raw Score: 0.3525|Loss: 0.3776 - Raw Score: 0.3354|Loss: 0.3767 - Raw Score: 0.3333|Loss: 0.3788 - Raw Score: 0.3285|Loss: 0.3784 - Raw Score: 0.3295|0.31968|0.30270|

## The use of class weight CE loss

Below experiment is done under the setting of one Deberta and six head, QA splited

|Stragegy|Fold0Epoch2(Train)|Fold1Epoch2(Train)|Fold2Epoch2(Train)|Fold3Epoch2(Train)|Fold4Epoch2(Train)|Public(Eval)|Private(Eval)|
|-|-|-|-|-|-|-|-|
|Without weight|Loss: 0.3730 - Raw Score: 0.3525|Loss: 0.3776 - Raw Score: 0.3354|Loss: 0.3767 - Raw Score: 0.3333|Loss: 0.3788 - Raw Score: 0.3285|Loss: 0.3784 - Raw Score: 0.3295|0.31968|0.30270|
|With weight|Loss: 0.3964 - Raw Score: 0.3555|Loss: 0.4015 - Raw Score: 0.3331|Loss: 0.3990 - Raw Score: 0.3412|Epoch 2 - Loss: 0.4013 - Raw Score: 0.3379|Epoch 2 - Loss: 0.3999 - Raw Score: 0.3431|||

## The use of larger lr for head

|Smaller LR|Loss: 0.3730 - Raw Score: 0.3525|Loss: 0.3776 - Raw Score: 0.3354|Loss: 0.3767 - Raw Score: 0.3333|Loss: 0.3788 - Raw Score: 0.3285|Loss: 0.3784 - Raw Score: 0.3295|0.31968|0.30270|
|Larger LR|Loss: 0.3894 - Raw Score: 0.3876|Loss: 0.3905 - Raw Score: 0.3744|Loss: 0.3913 - Raw Score: 0.3759|Loss: 0.3911 - Raw Score: 0.3653|Epoch 2 - Loss: 0.3906 - Raw Score: 0.3718|||