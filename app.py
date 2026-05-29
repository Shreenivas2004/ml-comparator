import json
from flask import request, Flask, url_for, redirect, render_template,session 
import pandas as pd 
import numpy as np
import os,shutil
from encoding import encoding,encoding_cluster
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,kneighbors_graph
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
import plotly.express as px
import plotly.utils as pu
import plotly.io as pio
app = Flask(__name__)
file_name="upload"
os.makedirs(file_name,exist_ok=True) 
app.config["SECRET_KEY"] = "Manu@2004"
@app.route('/',methods=["GET","POST"])
def data():
    if request.method=="POST":
        opt = request.form["option"]
        if opt=="Classification":
            return redirect(url_for("result", opt=opt))
        else:
            return redirect(url_for("result",opt=opt))    
    return render_template("home.html")



@app.route('/result/<opt>', methods=["GET", "POST"])
def result(opt):
    option = {"classification":["Decsion Tree","Logistic Regression","Navies Bayes","KNN","Support Vector Machine","All"],
              "clustering":["K Means","hiearchial","All"]}
    if opt=="Classification": 
        options = option["classification"]
        if request.method=="POST":
            df1 = request.files["Dataset"]
            session["data_name"] = df1.filename
            df1.save(f"{file_name}/{secure_filename(df1.filename)}")
            session["model_name"] = request.form.get("model_name")
            return redirect(url_for("model"))
    elif opt == "Clustering":
        options = option["clustering"]
        if request.method=="POST":
            df1 = request.files["Dataset"]
            session["data_name"] = df1.filename
            df1.save(f"{file_name}/{secure_filename(df1.filename)}")
            session["model_name"] = request.form.get("model_name")
            session["clusters"] = request.form.get("n_clusters")
            return redirect(url_for("cluster"))

    
    return render_template("result.html", opt=opt,options=options)



@app.route("/models", methods=["POST","GET"])
def model():
    # variables
    report=None
    accuracy=0
    results=None
    reports=None
    RocAucScore= 0
    f1Score = 0
    F1Score = None
    RocaucScore = None
    Accuracy = None
    df1 = session.get("data_name")
    file,file_extension=os.path.splitext(f"{file_name}/{df1}")
    model_name = session.get("model_name")
    model_keys = {"Decsion Tree":DecisionTreeClassifier(),"Logistic Regression":LogisticRegression(),"Navies Bayes":GaussianNB(),"KNN":KNeighborsClassifier(),"Support Vector Machine":SVC(),"All":["Decsion Tree","Logistic Regression","Navies Bayes","KNN"]}
    # reading dataset
    if file_extension==".xlsx" or file_extension==".csv":
        if file_extension==".csv":
            df = pd.read_csv(f"{file}{file_extension}")
        else:
            df = pd.read_excel(f"{file}{file_extension}")

    else:
        return f"plz upload an xlsx or csv file only"
    df_copy = df
    os.remove(f"{file}{file_extension}")
    
    #preprocessing
    x,y = df.iloc[:,:-1],df.iloc[:,-1]
    x,y = encoding(x,y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

    # training and testing
    if model_name != "All":
        model = model_keys[model_name]
        params={"random_state":1}
        if model_name == "Decsion Tree" or model_name=="Logistic Regression":
            model.set_params(**params)
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred=preds)
        f1Score = f1_score(y_test,preds,average="weighted")
        report = f"{classification_report(y_test,preds)}"
    else:
        models = model_keys[model_name]
        Accuracy = {}
        F1Score = {}
        RocaucScore={}
        reports = {}
        for i in models:
            model = model_keys[i]
            params={"random_state":1}
            if i == "Decsion Tree" or i=="Logistic Regression":
                model.set_params(**params)
            model.fit(x_train,y_train)
            preds = model.predict(x_test)
            Accuracy[i] = round(accuracy_score(y_test,y_pred=preds),2)
            F1Score[i] = round(f1_score(y_test,preds,average="weighted"),2) 
            reports[i] = f"{classification_report(y_test,preds)}"

    return render_template("model.html",
                            methods=["GET","POST"], 
                            model=model_name, 
                            df1=df1,file=file,
                            file_extension=file_extension, 
                            report=report,
                            accuracy=round(accuracy,2), 
                            results=results, 
                            reports=reports, 
                            f1Score=round(f1Score,2),
                            F1Score=F1Score,
                            Accuracy=Accuracy)


@app.route("/clustering", methods=["GET","POST"])
def cluster():
    # reading the dataset 
    pio.templates.default = "presentation"
    df1 = session.get("data_name")
    df1 = df1.replace(" ", "_")
    model_name = session.get("model_name")
    clusters = session.get("clusters")
    file,file_extension=os.path.splitext(f"{file_name}/{df1}")
    if file_extension==".xlsx" or file_extension==".csv":
        if file_extension==".csv":
            df = pd.read_csv(f"{file}{file_extension}")
        else:
            df = pd.read_excel(f"{file}{file_extension}")
    else:
        return f"plz upload an xlsx or csv file only"
    df_copy = df
    df = encoding_cluster(df)
    model_name = session.get("model_name")
    model_keys = {"K Means":KMeans(),"hiearchial":AgglomerativeClustering(),"All":["K Means","hiearchial"]}
    graphs = {}
    columns = df.columns.tolist()
    x,y=None,None
    if request.method=="POST":
        x_data = request.form.get("x_axis")
        y_data=  request.form.get("y_axis")
    else:
        x,y = df.iloc[:,0], df.iloc[:,1]
    if model_name != "All":
        model = model_keys[model_name]
        params={"n_clusters":int(clusters)}
        model.set_params(**params)
        model.fit(df)
        labels = model.labels_
        graph = px.scatter(x = x,y = y,color=labels.astype(str), labels={"x":x.name,"y":y.name})
        graphjson = json.dumps(graph,cls = pu.PlotlyJSONEncoder)
        return render_template("cluster.html",  graphjson = graphjson, columns=columns, model = model_name )
    else:
        models = model_keys[model_name]
        for i in models:
            model = model_keys[i]
            params={"n_clusters":int(clusters)}
            model.set_params(**params)
            model.fit(df)
            labels = model.labels_
            if x is not None and y is not None:
                graph = px.scatter(x = x,y = y,color=labels.astype(str), labels={"x":x.name,"y":y.name})
                graphjson = json.dumps(graph,cls = pu.PlotlyJSONEncoder)
                graphs[i] = graphjson
            if request.method == 'POST':
                graphs.clear()
                for i in models:
                    graph = px.scatter(x = df[x_data],y = df[y_data],color=labels.astype(str),labels={"x":x_data,"y":y_data} )
                    graphjson = json.dumps(graph,cls = pu.PlotlyJSONEncoder)
                   
                    graphs[i] = graphjson
        return render_template("cluster.html",  graphs = graphs, columns=columns,model = model_name)
            


if __name__ == "__main__":
    app.run(debug=True)

