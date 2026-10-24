import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json



def graph_metrics(results,name):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot vertical bars
    bars = ax.bar(results.keys(), results.values(), color='skyblue', edgecolor='black', width=0.6)
    ax.bar_label(bars, padding=3)
    # Customizations
    ax.set_title(f'{name} - metrics')
    ax.set_xlabel("Metrics")
    ax.set_ylabel('Count')

    plt.show()

def display_preds(preds, name):
    plt.figure(figsize=(12, 7))
    plt.bar(samples.keys(), preds, color=["#00bfff", "#00bfff", "#ffe600", "#ffe600","#00bfff", "#00bfff", "#ffe600", "#ffe600"])
    plt.ylabel(f"Predicted Tumor Fraction - {name}")
    plt.title(f"Predicted Tumor Fraction for Each Sample - {name}")  
    plt.show()
    plt.savefig(f"predicted_tumor_fraction_barplot_{name}.png")
    plt.close()





#______________________________________________________________________________________________________________________________________________________________________________



for name in range(1,11,2):
    lr_metrics = {"um_to_blood":[], "cancer_to_blood":[], "um_to_cancer":[], "synthetic_error":[]}
    #start_logistic_regression
    print(name*50)
    y = np.load(f'output_matrix_top_{name*50}_probes.npy')
    X = np.load(f'feature_matrix_top_{name*50}_probes.npy')

    #k-fold cross validation: 

    model = sklearn.linear_model.LogisticRegression()


    #train start
    print("Training model on full data...")
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_x = scaler.transform(X)
    model.fit(scaled_x, y)
    #train end

    #get the values  
    path = f"top_{name*50}_probes_probe_beta_pairs.csv"
    df = pd.read_csv(path)
    ref_probes = df["probe"].tolist()
    samples = {}
    with open("pilot_8samples_merged.csv", 'r') as file:
        #create dictionary, sample_wise probe addition
        file = file.read().splitlines()
        labels = file[0].split(",")
        for i in file[1:]:
            line = i.split(',')
            if line[0] in ref_probes:
                for j in range(1,len(line)):
                    if labels[j] in samples:
                        samples[labels[j]].append(float(line[j]))
                    else:
                        samples[labels[j]] = [float(line[j])]


    #inference
    preds = [x[1] for x in model.predict_proba(scaler.transform([samples[i] for i in samples]))]
    display_preds(preds, f"logistic regression - {name*50} probes")

    
    um_to_blood = ((preds[6]+preds[7])/2) /((preds[2]+preds[3])/2)
    cancer_to_blood = ((preds[0]+preds[1])/2) /((preds[2]+preds[3])/2)
    um_to_cancer = um_to_blood / cancer_to_blood
    lr_metrics = {"um_to_blood":um_to_blood, "cancer_to_blood":cancer_to_blood, "um_to_cancer":um_to_cancer}
    print(lr_metrics)

    graph_metrics(lr_metrics, f"logistic regression - {name*50} probes")

    #synthetic test: does the percentage drop as synthetic retina/retinoblastoma mix happens? 



    print("\n")
    coef = model.coef_
    intercept = model.intercept_

    coeff_probes = {x:y for x, y in zip(ref_probes, list(coef[0]))}
    max_coef = max(coef[0])
    max_item = max(coeff_probes, key = lambda x: coeff_probes[x])

    print("\n")
    print(max_item)
    print(max_coef)

    min_coef = min(coef[0])
    min_item = min(coeff_probes, key = lambda x: coeff_probes[x])
    print("\n")
    print(min_item)
    print(min_coef)

    betas = pd.read_csv(f"top_{50*name}_probes_probe_beta_pairs.csv")
    beta_vals = np.array([list(betas["beta_rb"])[list(betas["probe"]).index(x)] for x in [x for x in coeff_probes.keys()] ])

    plt.scatter(beta_vals, coef, color='blue', marker='o')
    plt.xlabel("beta")
    plt.ylabel("coef")
    plt.title("Coefficients of Model vs Beta")
    m, b = np.polyfit(beta_vals, coef[0], 1)  # Calculate slope and intercept
    plt.plot(beta_vals, m*beta_vals + b, color='green', label=f'Trendline (y={m:.2f}x+{b:.2f})')
    plt.legend()
    plt.show()


    #save model and scaler


    data = {
        "probes":ref_probes, 
        "scaler_var": scaler.var_.tolist(), 
        "scaler_mean": scaler.mean_.tolist(),
        "model_coef": {x:y for x, y in zip(ref_probes, list(coef[0]))},
        "model_int": model.intercept_.tolist()
    }

    with open(f"model_and_scaler_{name*50}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)



