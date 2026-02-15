from feature_extraction import FeatureExtractor,load_dataset,split_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import json
import time

def evaluate_model(y_true,y_pred):
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,average='weighted')
    recall=recall_score(y_true,y_pred,average='weighted')
    f1=f1_score(y_true,y_pred,average='weighted')
    cm=confusion_matrix(y_true,y_pred)
    
    return {
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'f1_score':f1,
        'confusion_matrix':cm.tolist()
    }

def run_experiments():
    print("Loading dataset...")
    texts,labels=load_dataset('dataset.json')
    
    print("Splitting data...")
    train_texts,train_labels,test_texts,test_labels=split_data(texts,labels)
    
    print("Training samples:",len(train_texts))
    print("Testing samples:",len(test_texts))
    
    extractor=FeatureExtractor()
    extractor.build_vocabulary(train_texts)
    extractor.build_ngram_vocabulary(train_texts,2)
    
    feature_methods={
        'bag_of_words':extractor.bag_of_words,
        'tfidf':extractor.tfidf,
        'bigrams':lambda docs: extractor.ngram_features(docs,2)
    }
    
    classifiers={
        'Naive Bayes':MultinomialNB(),
        'Decision Tree':DecisionTreeClassifier(random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=100,random_state=42)
    }
    
    results={}
    
    for feature_name,feature_func in feature_methods.items():
        print("\n"+"="*50)
        print("Feature:",feature_name)
        print("="*50)
        
        start_time=time.time()
        train_features=feature_func(train_texts)
        test_features=feature_func(test_texts)
        feature_time=time.time()-start_time
        
        print("Feature extraction time: {:.2f} seconds".format(feature_time))
        print("Feature vector size:",len(train_features[0]))
        
        results[feature_name]={}
        
        for clf_name,classifier in classifiers.items():
            print("\n  Classifier:",clf_name)
            
            start_time=time.time()
            classifier.fit(train_features,train_labels)
            train_time=time.time()-start_time
            
            predictions=classifier.predict(test_features)
            
            metrics=evaluate_model(test_labels,predictions)
            metrics['training_time']=train_time
            metrics['feature_extraction_time']=feature_time
            
            print("    Accuracy: {:.4f}".format(metrics['accuracy']))
            print("    Precision: {:.4f}".format(metrics['precision']))
            print("    Recall: {:.4f}".format(metrics['recall']))
            print("    F1-Score: {:.4f}".format(metrics['f1_score']))
            print("    Training Time: {:.4f}s".format(train_time))
            
            results[feature_name][clf_name]=metrics
    
    with open('results.json','w') as f:
        json.dump(results,f,indent=2)
    
    print("\n"+"="*50)
    print("Results saved to results.json")
    print("="*50)
    
    return results

if __name__=='__main__':
    results=run_experiments()