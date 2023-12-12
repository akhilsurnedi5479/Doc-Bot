import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
from flask import Flask, render_template, jsonify, request, session
import json
import uuid

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

# save data
data = {"users": []}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)


def write_json(new_data, filename='DATA.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


df_tr = pd.read_csv('Medical_dataset/Training.csv')
df_tt = pd.read_csv('Medical_dataset/Testing.csv')

symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])

# # I- GET ALL SYMPTOMS

all_symp_col = list(df_tr.columns[:-1])


def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow')


all_symp = [clean_symp(sym) for sym in (all_symp_col)]


def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)


all_symp_pr = [preprocess(sym) for sym in all_symp]

# associate each processed symp with column name
col_dict = dict(zip(all_symp_pr, all_symp_col))


# II- Syntactic Similarity

# Returns all the subsets of a set. This is a generator.
# {1,2,3}->[{},{1},{2},{3},{1,3},{1,2},..]
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


# Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a


# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])


# check if a txt and all diferrent combination if it exists in processed symp list
def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        # print(permutations(comb))
        for sym in permutations(comb):
            if sym in all_symp_pr:
                # print(sym)
                return sym
    return False


# Jaccard similarity 2docs
def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# apply vanilla jaccard to symp with all corpus
def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


# check a pattern if it exists in processed symp list
def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


# III- Semantic Similarity


from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def WSD(word, context):
    sens = lesk(context, word)
    return sens


# semantic similarity 2docs
def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                # x=syn1.path_similarity((syn2))
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


# apply semantic simarity to symp with all corpus
def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


# given a symp suggest possible synonyms
def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


# One-Hot-Vector dataframe
def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)


def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a


# list of symptoms --> possible diseases
def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis


# disease --> all symptoms
def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()


# IV- Prediction Model (KNN)
# load model
knn_clf = joblib.load('model/knn.pkl')

# ##  VI- SEVERITY / DESCRIPTION / PRECAUTION
# get dictionaries for severity-description-precaution for all diseases

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


# load dictionaries
getSeverityDict()
getprecautionDict()
getDescription()


# calcul patient condition
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary.keys():
            sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp)) > 13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")


# print possible symptoms
def related_sym(psym1):
    s = "could you be more specific, <br>"
    i = len(s)
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0

@app.route("/get_booking_details")
def get_booking_details():
    id = request.args.get('msg')
    booking_details = get_booking_by_id(id)
    return "testing dthe flow from py to html ---  "+id

def generate_numeric_id():
    # Generate a random 8-digit number
    return uuid.uuid4().int % 100000000

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return None


def write_json_file(file_path, data):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"Data written to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def make_a_booking(doctor_name, patient_name, patient_email, patient_age, patient_gender, selected_day, selected_time):
    doctor_file_path = "data/doctors.json"
    bookings_file_path = "data/bookings.json"
    # Load doctor data from the JSON file
    booking_id = generate_numeric_id()
    doctor_data = read_json_file(doctor_file_path)


    if not doctor_data:
        return

    schedule = doctor_data.get("schedule", {})

    if selected_day.lower() not in schedule:
        print("Invalid day selection.")
        return

    selected_day_schedule = schedule[selected_day.lower()]

    if selected_time not in selected_day_schedule:
        print("Invalid time selection.")
        return

    print(f"Appointment scheduled with {doctor_name} on {selected_day.capitalize()} at {selected_time}.")

    # Remove the selected time from the schedule
    selected_day_schedule.remove(selected_time)

    # Update the doctor JSON file with the modified schedule
    write_json_file(doctor_file_path, doctor_data)

    # Add booking details to the bookings JSON file
    booking_details = {
        "booking_id": booking_id,
        "doctor_name": doctor_name,
        "patient_name":patient_name,
        "patient_email": patient_email,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "appointment_day": selected_day.capitalize(),
        "appointment_time": selected_time
    }

    existing_bookings = read_json_file(bookings_file_path) or []
    existing_bookings.append(booking_details)

    write_json_file(bookings_file_path, existing_bookings)

    return booking_id
def get_booking_by_id(booking_id):
    try:
        with open('data/bookings.json', 'r') as file:
            bookings = json.load(file)
            for booking in bookings:
                if booking['booking_id'] == booking_id:
                    return booking
    except FileNotFoundError:
        print("Error: The 'bookings.json' file was not found.")
    return None

def modify_schedule_as_string(schedule):
    modified_string = ""

    for day, timings in schedule["schedule"].items():
        modified_string += f"{day.capitalize()}:<br>\n"
        if timings:
            for time in timings:
                modified_string += f"  - {time}<br>\n"
        else:
            modified_string += "  - No available times<br>\n"

    return modified_string


def get_available_schedule():
    doctor_file_path = "data/doctors.json"
    doctor_data = read_json_file(doctor_file_path)
    if not doctor_data:
        return



    #doctor_schedule = doctor_data.get("schedule", {})
    modified_schedule_string = modify_schedule_as_string(doctor_data)

    return modified_schedule_string


@app.route('/process_data', methods=['POST'])
def process_data():
    data_from_frontend = request.get_json()
    # Process the data in Python
    processed_data = {'message': f"Received from frontend: {data_from_frontend['data']}"}
    # get_booking_details_from_DB(bookingID)

    searchResult = get_booking_by_id(int(data_from_frontend['data']))
    print(searchResult)
    return jsonify(searchResult)


@app.route("/test")
def get_booking():
    return render_template("test.html")


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    s = request.args.get('msg')
    if "step" in session:
        if session["step"] == "Q_C":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if s == "q":
                "Thank you for using ower web site Mr/Ms " + name
            else:
                session["step"] = "FS"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender
    if s.upper() == "OK":
        return "What is your name ?"
    if 'name' not in session and 'step' not in session:
        session['name'] = s
        session['step'] = "email"
        return "Please enter your email ID"
    if session["step"] == "email":
        session["email"] = s
        session["step"]  = "age"
        return "How old are you? "

    if session["step"] == "age":
        session["age"] = int(s)
        session["step"] = "gender"
        return "Can you specify your gender ?"
    if session["step"] == "gender":
        session["gender"] = s
        session["step"] = "Depart"
    if session['step'] == "Depart":
        session['step'] = "BFS"
        return "Well, Hello again Mr/Ms " + session[
            "name"] + ", now I will be asking some few questions about your symptoms to see what you should do. Tap S to start diagnostic!"
    if session['step'] == "BFS":
        session['step'] = "FS"  # first symp
        return "Can you precise your main symptom Mr/Ms " + session["name"] + " ?"
    if session['step'] == "FS":
        sym1 = s
        sym1 = preprocess(sym1)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        temp = [sym1, sim1, psym1]
        session['FSY'] = temp  # info du 1er symptome
        session['step'] = "SS"  # second symptomee
        if sim1 == 1:
            session['step'] = "RS1"  # related_sym1
            s = related_sym(psym1)
            if s != 0:
                return s
        else:
            return "You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "RS1":
        temp = session['FSY']
        psym1 = temp[2]
        psym1 = psym1[int(s)]
        temp[2] = psym1
        session['FSY'] = temp
        session['step'] = 'SS'
        return "You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "SS":
        sym2 = s
        sym2 = preprocess(sym2)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        temp = [sym2, sim2, psym2]
        session['SSY'] = temp  # info du 2eME symptome(sym,sim,psym)
        session['step'] = "semantic"  # face semantic
        if sim2 == 1:
            session['step'] = "RS2"  # related sym2
            s = related_sym(psym2)
            if s != 0:
                return s
    if session['step'] == "RS2":
        temp = session['SSY']
        psym2 = temp[2]
        psym2 = psym2[int(s)]
        temp[2] = psym2
        session['SSY'] = temp
        session['step'] = "semantic"
    if session['step'] == "semantic":
        temp = session["FSY"]  # recuperer info du premier
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            session['step'] = 'PD'  # to possible_diseases
    if session['step'] == "BFsim1=0":
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            temp = []
            temp.append(sym1)
            temp.append(sim1)
            temp.append(psym1)
            session['FSY'] = temp
            session['step'] = "sim1=0"  # process of semantic similarity=1 for first sympt.
        else:
            session['step'] = "BFsim2=0"
    if session['step'] == "sim1=0":  # semantic no => suggestion
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if s == "yes":
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["FSY"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"] = suggest_syn(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                msg = "are you experiencing any  " + sugg[0] + "?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"
    if session['step'] == "BFsim2=0":
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            temp = []
            temp.append(sym2)
            temp.append(sim2)
            temp.append(psym2)
            session['SSY'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"
    if session['step'] == "sim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if s == "yes":
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["SSY"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"] = suggest_syn(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = "Are you experiencing " + sugg[0] + "?"
                session["suggested_2"] = sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step'] = "TEST"  # test if semantic and syntaxic and suggestion not found
    if session['step'] == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if sim1 == 0 and sim2 == 0:
            # GO TO THE END
            result = None
            session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'  # to possible_diseases
    if session['step'] == 'PD':
        # MAYBE THE LAST STEP
        # create patient symp list
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        print("hey2")
        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]
            print(session["all"])
        session["diseases"] = possible_diseases(session["all"])
        print(session["diseases"])
        all_sym = session["all"]
        diseases = session["diseases"]
        dis = diseases[0]
        session["dis"] = dis
        session['step'] = "for_dis"
    if session['step'] == "DIS":
        if "symv" in session:
            if len(s) > 0 and len(session["symv"]) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if s == "yes":
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                    print(possible_diseases(session["all"]))
                del symts[0]
                session["symv"] = symts
        if "symv" not in session:
            session["symv"] = symVONdisease(df_tr, session["dis"])
        if len(session["symv"]) > 0:
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(symts[0])
                session["asked"] = asked
                symts = session["symv"]
                msg = "Are you experiencing " + clean_symp(symts[0]) + "?"
                return msg
            else:
                del symts[0]
                session["symv"] = symts
                s = ""
                print("HANAAA")
                return get_bot_response()
        else:
            PD = possible_diseases(session["all"])
            diseases = session["diseases"]
            if diseases[0] in PD:
                session["testpred"] = diseases[0]
                PD.remove(diseases[0])
            #            diseases=session["diseases"]
            #            del diseases[0]
            session["diseases"] = PD
            session['step'] = "for_dis"
    if session['step'] == "for_dis":
        diseases = session["diseases"]
        if len(diseases) <= 0:
            session['step'] = 'PREDICT'
        else:
            session["dis"] = diseases[0]
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            return get_bot_response()  # turn around sympt of dis
        # predict possible diseases
    if session['step'] == "PREDICT":
        result = knn_clf.predict(OHV(session["all"], all_symp_col))
        session['step'] = "END"
    if session['step'] == "END":
        if result is not None:
            if result[0] != session["testpred"]:
                session['step'] = "Q_C"
                return "as you provide me with few symptoms, I am sorry to announce that I cannot predict your " \
                       "disease for the moment!!! <br> Can you specify more about what you are feeling or Tap q to " \
                       "stop the conversation "
            session['step'] = "Description"
            session["disease"] = result[0]
            return "Well Mr/Ms " + session["name"] + ", you may have " + result[
                0] + ". Tap D to get a description of the disease ."
        else:
            session['step'] = "Q_C"  # test if user want to continue the conversation or not
            return "can you specify more what you feel or Tap q to stop the conversation"
    if session['step'] == "Description":
        y = {"Name": session["name"], "Age": session["age"], "Gender": session["gender"], "Disease": session["disease"],
             "Sympts": session["all"]}
        write_json(y)
        session['step'] = "Severity"
        if session["disease"] in description_list.keys():
            return description_list[session["disease"]] + " \n <br>  How many days have you had symptoms?"
        else:
            if " " in session["disease"]:
                session["disease"] = session["disease"].replace(" ", "_")
            return "please visit <a href='" + "https://en.wikipedia.org/wiki/" + session["disease"] + "'>  here  </a>"
    if session['step'] == "Severity":
        # Code Change Start
        #session['step'] = 'FINAL'
        session['step']= 'DOCTOR'

        #Code change END
        if calc_condition(session["all"], int(s)) == 1:
            return "you should take the consultation from doctor. Do you want to make a booking? yes or no?"
        else:
            msg = 'Nothing to worry about, but you should take the following precautions :<br> '
            i = 1
            for e in precautionDictionary[session["disease"]]:
                msg += '\n ' + str(i) + ' - ' + e + '<br>'
                i += 1
            msg += ' Tap q to end'
            return msg
    # #Code Change START
    if session['step'] == "DOCTOR":

        #session['step'] = 'FINAL'
        if s.lower() == "yes":
            booking_day = ""
            booking_time = ""

            session['step'] = "SCHEDULE_SENT_TO_USER"
            schedule_available = get_available_schedule()

            #get_available_schedule()
            return "the available timings are:......Select the one you meant...<br>select '0'<br>"+str(schedule_available)
        else:
            session['step'] = 'FINAL'

    if session['step'] == "SCHEDULE_SENT_TO_USER":
        session['step'] = "FINAL"
        #if(int(s) == 0):
        schedule_string =  s
        words = schedule_string.split()

        # Assuming the day is the first word and the time is the second word
        day = words[0]
        time = words[1] + " " + words[2]

        print("Day:", day)
        print("Time:", time)
        #make a booking for schedule 0
        doctor_name = "Dr. John Smith"
        patient_name = session["name"]
        patient_email = session["email"]
        patient_age = session["age"]
        patient_gender = session["gender"]
        booking_day = day
        booking_time =time
        booking_id = make_a_booking(doctor_name, patient_name, patient_email, patient_age,patient_gender,booking_day,booking_time)
        return "Your booking has been made. Please note your booking id: "+str(booking_id)


    # #Code change END
    if session['step'] == "FINAL":
        session['step'] = "BYE"
        return "Your diagnosis was perfectly completed. Do you need another medical consultation (yes or no)? "
    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        email = session["email"]
        session.clear()
        if s.lower() == "yes":
            session["gender"] = gender
            session["email"] = email
            session["name"] = name
            session["age"] = age
            session['step'] = "FS"
            return "HELLO again Mr/Ms " + session["name"] + " Please tell me your main symptom. "
        else:
            return "THANKS Mr/Ms " + name + " for using me for more information please contact <b> +21266666666</b>"


if __name__ == "__main__":
    import random  # define the random module
    import string

    S = 10  # number of characters in the string.
    # call random.choices() string module to find the string in Uppercase + numeric data.
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
    # chat_sp()
    app.secret_key = str(ran)
    app.run(host="0.0.0.0")
