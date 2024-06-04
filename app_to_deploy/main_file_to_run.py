import streamlit as st
from annotated_text import annotated_text
from app_stage.run import run, model_init
from app_sbn.run import test, sbn_models_init
from prepare_text import prepare_text
import ast

text_box = st.form('users_text')

st.write("Choose model types:")

stage_model_use = st.checkbox('STAGE', value=False)
sbn_model_use = st.checkbox('SBN', value=False)



sentence = text_box.text_area('Your comment:', height=120)
submit = text_box.form_submit_button(f'Make triplets!')


sentence = prepare_text(sentence)

if len(sentence) == 0:
    st.write("Empty comment")

palette = {
    'POS':[
        "#45ff9d",
        "#5effaa",
        "#78ffb8",
        "#91ffc5",
        "#abffd2",
        "#c4ffe0", 
        "#deffed",
        "#1de27a",
        "#34e587",
        "#4ae895",
        "#61eba2",
        "#78eeaf",
        "#8ef1bd",
        "#a5f4ca"
    ],
    'NEG':[
        "#be3434",
        "#cb4141",
        "#d15555",
        "#d66969",
        "#db7d7d",
        "#e19191",
        "#e6a5a5",
        "#c13e56",
        "#c13e4b",
        "#c13e40",
        "#be3434"
    ],
    'NEU':[
        "#9dcefe",
        "#9dc6fe",
        "#9db6fe",
        "#9daefe",
        "#9da6fe",
        "#9d9efe",

    ]
}

if stage_model_use:
    stage_model = model_init()

if sbn_model_use:
    gcn_model, Bert, step_1, step_2_forward, step_2_reverse = sbn_models_init()

if submit:
    st.write('STAGE prediction:')
    stage_preds = None
    if stage_model_use:
        stage_preds = run("{}####[]\n".format(sentence), stage_model)
    if stage_preds is None:
        st.write("No triplets are found by STAGE :(")
    else:
        sent = sentence.split(" ")
        aspect_list = set()
        opinion_list = set()
        link_i = {
            'POS': 0,
            'NEG': 0,
            'NEU': 0
        }
        neg_i = 0
        for pred in stage_preds[0]:
            for i in range(pred[0][0], pred[0][1] + 1):
                if i in aspect_list:
                    sent[i][1] += str(link_i[pred[2]])
                    continue
                sent[i] = [sent[i], 'A_' + str(link_i[pred[2]]), palette[pred[2]][link_i[pred[2]]]]
                aspect_list.add(i)
            for i in range(pred[1][0], pred[1][1] + 1):
                if i in opinion_list:
                    sent[i][1] += str(link_i[pred[2]])
                    continue
                sent[i] = [sent[i], 'O_' + str(link_i[pred[2]]), palette[pred[2]][link_i[pred[2]]]]
                opinion_list.add(i)
            link_i[pred[2]] += 1

        new_sent = []
        for i in sent:
            if type(i) is list:
                new_sent.append(tuple(i))
            else:
                new_sent.append(i)
            new_sent.append(" ")
        print(new_sent)
        annotated_text(new_sent)

    st.write('SBN prediction:')
    sbn_preds = None
    if sbn_model_use:
        sbn_preds, bert_tokens = test(["{}####[([0], [1], 'STR')]\n".format(sentence)], gcn_model, Bert, step_1, step_2_forward, step_2_reverse)
        print(sbn_preds)

    if sbn_preds is None:
        st.write("No triplets are found by SBN :(")
    else:
        sent = bert_tokens
        aspect_list = set()
        opinion_list = set()
        link_i = {
            'POS': 0,
            'NEG': 0,
            'NEU': 0
        }
        neg_i = 0
        for pred_0 in sbn_preds:
            pred = ast.literal_eval(pred_0)
            print(pred)
            for i in range(pred[0][0], pred[0][1]):
                if i in aspect_list:
                    sent[i][1] += str(link_i[pred[2]])
                    continue
                sent[i] = [sent[i], 'A_' + str(link_i[pred[2]]), palette[pred[2]][link_i[pred[2]]]]
                aspect_list.add(i)
            for i in range(pred[1][0], pred[1][1]):
                if i in opinion_list:
                    sent[i][1] += str(link_i[pred[2]])
                    continue
                sent[i] = [sent[i], 'O_' + str(link_i[pred[2]]), palette[pred[2]][link_i[pred[2]]]]
                opinion_list.add(i)
            link_i[pred[2]] += 1

        new_sent = []
        for i in sent[1:-1]:
            if type(i) is list:
                if '##' not in i[0]:
                    new_sent.append(tuple(i))
                    new_sent.append(" ")
                else:
                    x = [new_sent[-2][0] + i[0][2:], new_sent[-2][1], new_sent[-2][2]]
                    print(x)
                    new_sent[-2] = tuple(x)
            else:
                if '##' in i:
                    new_sent[-2] += i[2:]
                else:
                    new_sent.append(i)
                    new_sent.append(" ")
        print(new_sent)
        annotated_text(new_sent)
