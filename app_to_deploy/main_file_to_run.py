import streamlit as st
from annotated_text import annotated_text
from app_stage.run import run
from prepare_text import prepare_text


text_box = st.form('users_text')

st.write("Choose model types:")
sbn_model_use = st.checkbox('SBN')
stage_model_use = st.checkbox('STAGE')
# using_model = st.multiselect("Model for use:", options=['SBN', 'STAGE'], default='STAGE')


sentence = text_box.text_area('Your comment:', height=150)
submit = text_box.form_submit_button(f'Make triplets!')


sentence = prepare_text(sentence)

if len(sentence) == 0:
    st.write("Empty comment")

with open('test_triplets.txt', 'w') as f:
    f.write("{}####[]\n".format(sentence))

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
        "#e6a5a5"
    ]
}

if submit:
    preds = run(sentence)
    if preds is None:
        st.write("No triplets are found :(")
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
        for pred in preds[0]:
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
            # a = ""
            # for i in pred[0][:-1]:
            #     a += sent[i]
            # b = ""
            # for i in pred[1][:-1]:
            #     b += sent[i]
            # st.write(a, b, pred[2])
        new_sent = []
        for i in sent:
            if type(i) is list:
                new_sent.append(tuple(i))
            else:
                new_sent.append(i)
            new_sent.append(" ")
        print(new_sent)
        annotated_text(new_sent)
