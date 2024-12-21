from flask import render_template, request, send_file, session, redirect, url_for, jsonify, send_from_directory
from app import app
from flask_cors import CORS
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import os
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import re
from sentence_transformers import SentenceTransformer, util
import torch
import shutil
import tempfile
from codecarbon import EmissionsTracker
import redis
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import io
import base64
from flask import Response
import plotly.express as px
import random
import numpy as np
import concurrent.futures

# Zorg voor deterministisch gedrag in PyTorch
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logging.basicConfig(level=logging.INFO)

ECLI_texts = {}
ECLI_cache = {}

# Stel de regio in via omgevingsvariabele
os.environ['CODECARBON_COUNTRY'] = 'US'  # Vervang 'US' door de juiste ISO-landcode

# Laad beide modellen
multi_qa_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
fine_tuned_model = SentenceTransformer('output/fine-tuned-model')
fine_tuned_model_dutch_legal = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')

# CORS configuratie
CORS(app)

@app.route('/clear_url')
def clear_url():
    session.pop('url', None)
    session.pop('unique_list', None)
    session.pop('file_path', None)
    return redirect(url_for('welcome'))

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        app.logger.info("Upload POST request received")
        if 'file' not in request.files:
            app.logger.error("No file part in the request")
            return jsonify({"status": "failed", "message": "No file part"})
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("No selected file")
            return jsonify({"status": "failed", "message": "No selected file"})

        try:
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                app.logger.info(f"Saving file to: {file_path}")
                file.save(file_path)
                
                # Additional processing if needed
                session['file_path'] = file_path
                unique_list = load_eclis_from_excel(file_path)
                session['unique_list'] = unique_list
                app.logger.info(f"Unique ECLI list: {unique_list}")

                return redirect(url_for('loading'))
        except Exception as e:
            app.logger.error(f"Error uploading file: {e}")
            return jsonify({"status": "failed", "message": "File upload failed"})
        
    return render_template('upload.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    unique_list = session.get('unique_list', [])
    app.logger.info(f"Loaded unique_list from session: {unique_list}")
    for ecli in unique_list:
        if ecli not in ECLI_cache:
            app.logger.info(f"Requesting data for ECLI: {ecli}")
            api_request(ecli)

        # Update de JSON-file na het voltooien van de API-requests
    update_json_file()
    return jsonify({"status": "done"})

@app.route('/choose_analysis')
def choose_analysis():
    return render_template('choose_analysis.html')

@app.route('/traditional_search', methods=['GET', 'POST'])
def traditional_search():
    if request.method == 'POST':
        search_terms = request.form.get('search_terms', '').strip()
        include_synonyms = 'include_synonyms' in request.form
        session['search_terms'] = search_terms
        session['include_synonyms'] = include_synonyms
        return redirect(url_for('traditional_analysis'))
    return render_template('traditional_search.html')

@app.route('/traditional_analysis', methods=['GET', 'POST'])
def traditional_analysis():
    global ECLI_texts
    search_results_count = 0
    used_synonyms = {}

    search_terms = session.get('search_terms', '')
    include_synonyms = session.get('include_synonyms', False)

    tracker = EmissionsTracker()  # Begin tracking CO2-uitstoot
    tracker.start()

    start_time = time.time()  # Start tijdmeting

    if search_terms:
        search_terms = [term.split('|') for term in search_terms.split(',') if term]

        if include_synonyms:
            extended_search_terms = []
            for term_group in search_terms:
                extended_group = set(term_group)
                for term in term_group:
                    synonyms = get_synonyms(term)
                    extended_group.update(synonyms)
                    if term not in used_synonyms:
                        used_synonyms[term] = set()
                    used_synonyms[term].update(synonyms)
                extended_search_terms.append(list(extended_group))
            search_terms = extended_search_terms

        ECLI_texts = {}

        unique_list = session.get('unique_list', [])
        for ecli in unique_list:
            root, identifier_link, date_link = api_request(ecli)
            if root is None:
                continue
            texts = [str(elem.text) for elem in root.iter() if elem.text and all(
                any(synonym.lower() in str(elem.text).lower() for synonym in term)
                for term in search_terms
            )]
            highlighted_texts = []
            for text in texts:
                for term_group in search_terms:
                    for synonym in term_group:
                        text = highlight_term(text, synonym)
                highlighted_texts.append(text)
            ECLI_texts[ecli] = {'texts': highlighted_texts, 'identifier_link': identifier_link, 'current_index': 0, 'date_link': date_link}
            search_results_count += len(highlighted_texts)

        session['used_synonyms'] = {term: list(synonyms) for term, synonyms in used_synonyms.items()}

    update_excel_file()
    update_json_file()

    scroll_position = session.get('scroll_position', 0)
    app.logger.info(f"ECLI_texts: {ECLI_texts}")  # Voeg logging toe om de inhoud van ECLI_texts te controleren

    end_time = time.time()  # Eind tijdmeting
    elapsed_time = end_time - start_time
    app.logger.info(f"Traditional Search tijd: {elapsed_time} seconden")

    emissions_traditional = tracker.stop()  # Stop tracking en krijg de uitstoot in kg
    app.logger.info(f"Traditional Search CO2-uitstoot: {emissions_traditional} kg")

    # Voeg de semantische zoekopdracht uitstoot toe als je die ook hebt
    emissions_semantic = session.get('emissions_semantic', 0)
    
    if emissions_semantic is None:
        emissions_semantic = 0

    emissions_data = [emissions_traditional, emissions_semantic]
    session['emissions_traditional'] = emissions_traditional  # Save the traditional search emissions in session

    session['current_analysis'] = 'traditional'  # Markeer dat traditionele analyse actief is

    return render_template('traditional_analysis.html', ECLI_texts=ECLI_texts, search_results_count=search_results_count, used_synonyms=used_synonyms, search_terms=search_terms, scroll_position=scroll_position, emissions_data=emissions_data)

@app.route('/previous_ecli/<ecli>', methods=['GET'])
def previous_ecli(ecli):
    logging.info(f"Previous ECLI called for: {ecli}")
    if ecli in ECLI_texts:
        current_index = ECLI_texts[ecli]['current_index']
        ECLI_texts[ecli]['current_index'] = max(0, current_index - 1)
        logging.info(f"Updated index for {ecli}: {ECLI_texts[ecli]['current_index']}")
    update_excel_file()
    update_json_file()

    text = ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']] if ECLI_texts[ecli]['texts'] else 'No results'
    return jsonify({
        'text': text,
        'index': ECLI_texts[ecli]['current_index'] + 1,
        'length': len(ECLI_texts[ecli]['texts'])
    })

@app.route('/next_ecli/<ecli>', methods=['GET'])
def next_ecli(ecli):
    logging.info(f"Next ECLI called for: {ecli}")
    if ecli in ECLI_texts:
        current_index = ECLI_texts[ecli]['current_index']
        ECLI_texts[ecli]['current_index'] = min(len(ECLI_texts[ecli]['texts']) - 1, current_index + 1)
        logging.info(f"Updated index for {ecli}: {ECLI_texts[ecli]['current_index']}")
    update_excel_file()
    update_json_file()

    text = ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']] if ECLI_texts[ecli]['texts'] else 'No results'
    return jsonify({
        'text': text,
        'index': ECLI_texts[ecli]['current_index'] + 1,
        'length': len(ECLI_texts[ecli]['texts'])
    })

@app.route('/delete_ecli/<ecli>', methods=['GET'])
def delete_ecli(ecli):
    logging.info(f"Delete ECLI called for: {ecli}")
    if ecli in ECLI_texts:
        current_index = ECLI_texts[ecli]['current_index']
        if ECLI_texts[ecli]['texts']:
            del ECLI_texts[ecli]['texts'][current_index]
            if current_index >= len(ECLI_texts[ecli]['texts']):
                ECLI_texts[ecli]['current_index'] = max(0, len(ECLI_texts[ecli]['texts']) - 1)
            ECLI_texts[ecli]['texts'] = [text for text in ECLI_texts[ecli]['texts'] if text]
        logging.info(f"Deleted text for {ecli}, remaining texts: {len(ECLI_texts[ecli]['texts'])}")
    update_excel_file()
    update_json_file()

    return jsonify({
        'text': 'No results' if not ECLI_texts[ecli]['texts'] else ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']],
        'index': ECLI_texts[ecli]['current_index'] + 1,
        'length': len(ECLI_texts[ecli]['texts'])
    })

@app.route('/choose_model', methods=['POST'])
def choose_model():
    model_choice = request.form.get('model_choice')
    session['model_choice'] = model_choice
    return redirect(url_for('semantic_search'))

@app.route('/semantic_repeated_comparison', methods=['POST'])
def semantic_repeated_comparison():
    results = []
    semantic_search_query = session.get('semantic_search_query', '')
    similarity_score = session.get('similarity_score', 0.7)
    model_choice = session.get('model_choice', 'fine-tuned-dutch-legal')

    # Selecteer het juiste model
    model = choose_model_by_name(model_choice)

    if semantic_search_query:
        for i in range(100):  # Voer 100 keer dezelfde vergelijking uit
            ECLI_texts = {}
            start_time = time.time()
            unique_list = session.get('unique_list', [])
            for ecli in unique_list:
                root, identifier_link, date_link = api_request(ecli)
                if root is None:
                    continue
                texts = [elem.text for elem in root.iter() if elem.text]
                encoded_results = model.encode(texts, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
                encoded_query = model.encode(semantic_search_query, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
                cosine_scores = util.pytorch_cos_sim(encoded_query, encoded_results).cpu().numpy()[0]
                sorted_results = sorted(zip(texts, cosine_scores), key=lambda x: x[1], reverse=True)
                ECLI_texts[ecli] = {'texts': sorted_results, 'identifier_link': identifier_link, 'date_link': date_link}
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            results.append({
                'run': i+1,
                'results': ECLI_texts,
                'timestamp': datetime.now(),
                'time_elapsed': elapsed_time
            })
            print(f'current iteration {i}')
    
    # Sla de resultaten op in een Excelbestand
    excel_file_path = save_results_to_excel(results)
    
    return send_file(excel_file_path, as_attachment=True, download_name='semantic_comparison_results.xlsx')

def choose_model_by_name(model_choice):
    if model_choice == 'pretrained':
        return multi_qa_model
    elif model_choice == 'fine-tuned':
        return fine_tuned_model
    elif model_choice == 'fine-tuned-dutch-legal':
        return fine_tuned_model_dutch_legal
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

def save_results_to_excel(results):
    data = []
    for result in results:
        for ecli, texts in result['results'].items():
            current_text, score = texts['texts'][0] if texts['texts'] else ('No results', 0)
            data.append({
                'Run': result['run'],
                'ECLI': ecli,
                'Timestamp': result['timestamp'],
                'Text': current_text,
                'Score': score,
                'Time Elapsed (s)': result['time_elapsed']
            })
    
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
        df.to_excel(temp_file.name, index=False)
        return temp_file.name


# Verbinding maken met Redis (standaard configuratie)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

@app.route('/semantic_search', methods=['GET', 'POST'])
def semantic_search():
    if request.method == 'POST':
        app.logger.info("Semantic search POST request received")
        semantic_search_query = request.form.get('semantic_search_query', '').strip()
        similarity_score = float(request.form.get('similarity_score', 0.7))
        model_choice = request.form.get('model_choice', 'fine-tuned-dutch-legal')  # Standaard naar het nieuwe fine-tuned model
        session['semantic_search_query'] = semantic_search_query
        session['similarity_score'] = similarity_score
        session['model_choice'] = model_choice  # Sla de keuze op in de sessie
        app.logger.info(f"Semantic search query: {semantic_search_query}, similarity score: {similarity_score}, model choice: {model_choice}")
        return redirect(url_for('semantic_analysis'))

    return render_template('semantic_search.html')

@app.route('/semantic_analysis', methods=['GET', 'POST'])
def semantic_analysis():
    global ECLI_texts
    search_results_count = 0

    semantic_search_query = session.get('semantic_search_query', '')
    similarity_score = session.get('similarity_score', 0.7)
    model_choice = session.get('model_choice', 'fine-tuned')

    # Selecteer het juiste model op basis van de keuze van de gebruiker
    try:
        if model_choice == 'pretrained':
            model = multi_qa_model
            model_name = 'Pre-trained SBERT'
        elif model_choice == 'fine-tuned':
            model = fine_tuned_model
            model_name = 'Fine-Tuned: "Deskundigenbenoeming"'
        elif model_choice == 'fine-tuned-dutch-legal':
            model = fine_tuned_model_dutch_legal
            model_name = 'Fine-Tuned: Dutch Legal Cases'
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")
        logging.info(f"Model chosen: {model_name}")
    except Exception as e:
        logging.error(f"Error selecting model: {e}")
        return jsonify({"status": "failed", "message": f"Error selecting model: {e}"})

    tracker = EmissionsTracker()  # Begin tracking CO2-uitstoot
    tracker.start()

    start_time = time.time()  # Start tijdmeting

    if semantic_search_query:
        ECLI_texts = {}
        unique_list = session.get('unique_list', [])
        for ecli in unique_list:
            root, identifier_link, date_link = api_request(ecli)
            if root is None:
                continue  # Ga verder met de volgende ECLI als er een fout optrad bij het ophalen van de data
            texts = [elem.text for elem in root.iter() if elem.text]
            if texts:
                try:
                    # Gebruik cache voor vectors
                    cache_key = f'vectors_{ecli}'
                    cached_vectors = get_cached_vectors(cache_key)
                    
                    if cached_vectors is not None:
                        encoded_results = cached_vectors
                        logging.info(f"Loaded vectors from cache for ECLI: {ecli}")
                    else:
                        # Encode texts and query, move them to GPU if available
                        encoded_results = model.encode(texts, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
                        # Cache the vectors
                        cache_vectors(cache_key, encoded_results)
                        logging.info(f"Computed and cached vectors for ECLI: {ecli}")
                    
                    encoded_query = model.encode(semantic_search_query, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')

                    # Compute cosine similarities and move the result back to CPU if needed
                    cosine_scores = util.pytorch_cos_sim(encoded_query, encoded_results)
                    resultaten_met_scores = list(zip(texts, cosine_scores.cpu().numpy()[0] if torch.cuda.is_available() else cosine_scores.numpy()[0]))

                    gesorteerde_resultaten = sorted(resultaten_met_scores, key=lambda x: x[1], reverse=True)
                    if gesorteerde_resultaten:
                        semantic_texts = [(text, score, semantic_search_query, model_name) for text, score in gesorteerde_resultaten if score >= similarity_score]
                    else:
                        semantic_texts = []
                    ECLI_texts[ecli] = {'texts': semantic_texts, 'identifier_link': identifier_link, 'current_index': 0, 'date_link': date_link}
                    search_results_count += len(semantic_texts)
                except Exception as e:
                    logging.error(f"Error processing ECLI {ecli} with model {model_name}: {e}")
                    continue
            else:
                ECLI_texts[ecli] = {'texts': [], 'identifier_link': identifier_link, 'current_index': 0, 'date_link': date_link}

        update_excel_file()
        update_json_file()

    scroll_position = session.get('scroll_position', 0)

    end_time = time.time()  # Eind tijdmeting
    elapsed_time = end_time - start_time
    logging.info(f"Semantic Search tijd: {elapsed_time} seconden")

    emissions_semantic = tracker.stop()  # Stop tracking en krijg de uitstoot in kg
    logging.info(f"Semantic Search CO2-uitstoot: {emissions_semantic} kg")

    # Voeg de traditionele zoekopdracht uitstoot toe als je die ook hebt
    emissions_traditional = session.get('emissions_traditional', 0)
    
    if emissions_traditional is None:
        emissions_traditional = 0

    emissions_data = [emissions_traditional, emissions_semantic]
    session['emissions_semantic'] = emissions_semantic  # Save the semantic search emissions in session

    session['current_analysis'] = 'semantic'  # Markeer dat semantische analyse actief is

    return render_template('semantic_analysis.html', ECLI_texts=ECLI_texts, search_results_count=search_results_count, semantic_search_query=semantic_search_query, similarity_score=similarity_score, scroll_position=scroll_position, emissions_data=emissions_data)

def cache_vectors(key, vectors):
    pickled_vectors = pickle.dumps(vectors)
    redis_client.set(key, pickled_vectors)

def get_cached_vectors(key):
    pickled_vectors = redis_client.get(key)
    if pickled_vectors is not None:
        return pickle.loads(pickled_vectors)
    return None

@app.route('/next_semantic/<ecli>', methods=['GET'])
def next_semantic(ecli):
    logging.info(f"Next ECLI called for: {ecli}")
    try:
        if ecli in ECLI_texts:
            current_index = ECLI_texts[ecli]['current_index']
            ECLI_texts[ecli]['current_index'] = min(len(ECLI_texts[ecli]['texts']) - 1, current_index + 1)
            logging.info(f"Updated index for {ecli}: {ECLI_texts[ecli]['current_index']}")
            response_data = {
                'text': ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][0] if ECLI_texts[ecli]['texts'] else 'No results',
                'index': int(ECLI_texts[ecli]['current_index'] + 1),
                'length': int(len(ECLI_texts[ecli]['texts'])),
                'score': float(ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][1]) if ECLI_texts[ecli]['texts'] else 0.0,
                'query': ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][2] if ECLI_texts[ecli]['texts'] else '',
                'model_name': ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][3] if ECLI_texts[ecli]['texts'] else ''
            }
            update_excel_file()
            update_json_file()
            return jsonify(response_data)
        return jsonify({'error': 'ECLI not found'})
    except Exception as e:
        logging.error(f"Error in next_semantic: {e}")
        return jsonify({'error': str(e)})

@app.route('/previous_semantic/<ecli>', methods=['GET'])
def previous_semantic(ecli):
    logging.info(f"Previous ECLI called for: {ecli}")
    try:
        if ecli in ECLI_texts:
            current_index = ECLI_texts[ecli]['current_index']
            ECLI_texts[ecli]['current_index'] = max(0, current_index - 1)
            logging.info(f"Updated index for {ecli}: {ECLI_texts[ecli]['current_index']}")
            response_data = {
                'text': ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][0] if ECLI_texts[ecli]['texts'] else 'No results',
                'index': int(ECLI_texts[ecli]['current_index'] + 1),
                'length': int(len(ECLI_texts[ecli]['texts'])),
                'score': float(ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][1]) if ECLI_texts[ecli]['texts'] else 0.0,
                'query': ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][2] if ECLI_texts[ecli]['texts'] else '',
                'model_name': ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][3] if ECLI_texts[ecli]['texts'] else ''
            }
            update_excel_file()
            update_json_file()
            return jsonify(response_data)
        return jsonify({'error': 'ECLI not found'})
    except Exception as e:
        logging.error(f"Error in previous_semantic: {e}")
        return jsonify({'error': str(e)})
    
def get_current_text_details(ecli):
    current_index = ECLI_texts[ecli]['current_index']
    current_text = ECLI_texts[ecli]['texts'][current_index][0]
    scores = []
    query = []
    models = []

    for text, score, q, model_name in ECLI_texts[ecli]['texts']:
        if text == current_text:
            scores.append(score)
            query.append(q)
            models.append(model_name)

    return current_text, scores, query, models


@app.route('/delete_semantic/<ecli>', methods=['GET'])
def delete_semantic(ecli):
    logging.info(f"Delete ECLI called for: {ecli}")
    try:
        if ecli in ECLI_texts:
            current_index = ECLI_texts[ecli]['current_index']
            if ECLI_texts[ecli]['texts']:
                del ECLI_texts[ecli]['texts'][current_index]
                if current_index >= len(ECLI_texts[ecli]['texts']):
                    ECLI_texts[ecli]['current_index'] = max(0, len(ECLI_texts[ecli]['texts']) - 1)
                ECLI_texts[ecli]['texts'] = [text for text in ECLI_texts[ecli]['texts'] if text]
                response_text = 'No results' if not ECLI_texts[ecli]['texts'] else ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][0]
                response_score = 0.0 if not ECLI_texts[ecli]['texts'] else float(ECLI_texts[ecli]['texts'][ECLI_texts[ecli]['current_index']][1])
                response_data = {
                    'text': response_text,
                    'index': int(ECLI_texts[ecli]['current_index'] + 1),
                    'length': int(len(ECLI_texts[ecli]['texts'])),
                    'score': response_score
                }
                logging.info(f"Deleted text for {ecli}, remaining texts: {len(ECLI_texts[ecli]['texts'])}")
                update_excel_file()
                update_json_file()
                return jsonify(response_data)
        return jsonify({'error': 'ECLI not found'})
    except Exception as e:
        logging.error(f"Error in delete_semantic: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_all_semantic/<ecli>')
def delete_all_semantic(ecli):
    global ECLI_texts
    if ecli in ECLI_texts:
        del ECLI_texts[ecli]  # Verwijder alle resultaten voor de gegeven ECLI
        update_excel_file()  # Werk het Excel-bestand bij
        update_json_file()   # Werk het JSON-bestand bij
        return jsonify({"status": "success"})
    return jsonify({"error": "ECLI not found"})

@app.route('/download_template')
def download_template():
    return send_from_directory(directory='static', path='ecli_template.xlsx', as_attachment=True)

@app.route('/download/excel', methods=['GET'])
def download_excel():
    current_analysis = session.get('current_analysis', None)
    if current_analysis == 'traditional':
        temp_excel_file = session.get('temp_excel_file', None)
    elif current_analysis == 'semantic':
        temp_excel_file = session.get('temp_excel_file_semantic', None)
    else:
        return "No Excel file generated", 404

    if temp_excel_file:
        return send_file(
            temp_excel_file,
            as_attachment=True,
            download_name='ECLI_results.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        return "No Excel file generated", 404
    
@app.route('/download/json', methods=['GET'])
def download_json():
    current_analysis = session.get('current_analysis', None)
    if current_analysis == 'traditional':
        temp_json_file = session.get('temp_json_file', None)
    elif current_analysis == 'semantic':
        temp_json_file = session.get('temp_json_file_semantic', None)
    else:
        return "No JSON file generated", 404

    if temp_json_file:
        return send_file(
            temp_json_file,
            as_attachment=True,
            download_name='ECLI_results.json',
            mimetype='application/json'
        )
    else:
        return "No JSON file generated", 404

import plotly.express as px
import pandas as pd

@app.route('/save_tensors')
def save_tensors_route():
    unique_list = session.get('unique_list', [])
    if not unique_list:
        return "No data to save", 404
    
    sample_texts = []
    for ecli in unique_list[:10]:  # Beperk tot de eerste 10 ECLIs voor snelheid
        root, identifier_link, date_link = api_request(ecli)
        if root is None:
            continue
        texts = [elem.text for elem in root.iter() if elem.text]
        sample_texts.extend(texts)
    
    # Controleer of er tekstdata aanwezig is
    if not sample_texts:
        return "No texts found to save", 404
    
    # Encode de teksten
    embeddings = model.encode(sample_texts, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Maak een dictionary om de embeddings en teksten op te slaan
    data_to_save = {
        'texts': sample_texts,
        'embeddings': embeddings.cpu()
    }
    
    # Sla de data op in een .pt-bestand
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
        torch.save(data_to_save, temp_file.name)
        return send_file(temp_file.name, as_attachment=True, download_name='embeddings.pt')

def highlight_term(text, term):
    if text is None:
        return text  # Return None if text is None
    if not isinstance(text, str):
        text = str(text)  # Convert to string if not already a string
    return re.sub(r'(?i)('+re.escape(term)+r')', r'<span class="highlight">\1</span>', text)

def update_excel_file():
    data = []
    for ecli, texts in ECLI_texts.items():
        date = texts.get('date_link', 'No date available')
        if texts['texts']:
            current_text, scores, query, models = get_current_text_details(ecli)
            result_text = remove_html_tags(current_text)
            # Format the scores with a comma instead of a period
            score = ', '.join([f"{s:.2f}".replace('.', ',') for s in scores])
            query = query if isinstance(query, str) else ', '.join(query)
            model = ', '.join(models)
        else:
            result_text = 'none'
            score = 'none'
            query = 'none'
            model = 'none'
        link = texts.get('identifier_link', 'No link available')

        data.append((date, ecli, link, result_text, score, query, model))

    df = pd.DataFrame(data, columns=['Date', 'ECLI', 'Link', 'Result', 'Score', 'Query', 'Model'])

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
        df.to_excel(temp_file.name, index=False)

    if session.get('current_analysis') == 'traditional':
        session['temp_excel_file'] = temp_file.name
    elif session.get('current_analysis') == 'semantic':
        session['temp_excel_file_semantic'] = temp_file.name
    session.modified = True

def update_json_file():
    data = []
    for ecli, temp_file_name in ECLI_cache.items():
        with open(temp_file_name, 'r', encoding='utf-8') as file:
            root = ET.parse(file).getroot()
            identifier_link = None
            date_link = None
            namespaces = {
                'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                'dcterms': "http://purl.org/dc/terms/",
            }

            for identifier_tag in root.findall('.//rdf:Description/dcterms:identifier', namespaces):
                if identifier_tag.text and identifier_tag.text.startswith('http'):
                    identifier_link = identifier_tag.text
                    break

            for date_tag in root.findall('.//rdf:Description/dcterms:issued', namespaces):
                if date_tag.text:
                    try:
                        date_link = datetime.strptime(date_tag.text, '%Y-%m-%d').date()
                    except ValueError:
                        app.logger.warning(f"Invalid date format for ECLI {ecli}: {date_tag.text}")
                        date_link = None

            texts = [elem.text for elem in root.iter() if elem.text]

            data.append({
                'ecli': ecli,
                'identifier_link': identifier_link,
                'date_link': date_link,
                'texts': texts
            })

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8') as temp_file:
        json.dump(data, temp_file, indent=4, default=str)
        if session.get('current_analysis') == 'traditional':
            session['temp_json_file'] = temp_file.name
        elif session.get('current_analysis') == 'semantic':
            session['temp_json_file_semantic'] = temp_file.name
    session.modified = True

def load_eclis_from_excel(file_path):
    try:
        logging.info(f"Loading ECLIs from file: {file_path}")
        df = pd.read_excel(file_path)
        if 'ECLI' not in df.columns:
            raise ValueError("Excel file must contain an 'ECLI' column.")
        eclis = df['ECLI'].dropna().tolist()
        logging.info(f"Successfully loaded {len(eclis)} ECLIs from file.")
        return eclis
    except Exception as e:
        logging.error(f"Error loading ECLIs from file: {e}")
        return []

def clean_ecli(ecli):
    try:
        cleaned_ecli = ecli.split(' ')[0].strip()
        if not cleaned_ecli.startswith('ECLI:'):
            raise ValueError(f"Invalid ECLI format: {ecli}")
        logging.info(f"Cleaned ECLI: {cleaned_ecli} from original ECLI: {ecli}")
        return cleaned_ecli
    except Exception as e:
        logging.error(f"Error cleaning ECLI {ecli}: {e}")
        return None



def api_request(ecli):
    cleaned_ecli = clean_ecli(ecli)
    namespaces = {
        'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        'dcterms': "http://purl.org/dc/terms/",
    }

    logging.info(f"Requesting data for ECLI: {cleaned_ecli}")

    if cleaned_ecli in ECLI_cache:
        temp_file_name = ECLI_cache[cleaned_ecli]
        logging.info(f"Loading ECLI from cache: {cleaned_ecli}")
        try:
            with open(temp_file_name, 'rb') as file:
                root = ET.parse(file).getroot()
        except Exception as e:
            logging.error(f"Error parsing cached XML for {cleaned_ecli}: {e}")
            print(f"Skipping ECLI due to parse error: {cleaned_ecli}")
            return None, None, None
    else:
        url = f"https://data.rechtspraak.nl/uitspraken/content?id={cleaned_ecli}"
        logging.info(f"Fetching ECLI from URL: {url}")
        try:
            response = requests.get(url, stream=True)
            logging.info(f"API Response status code: {response.status_code}")
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as temp_file:
                shutil.copyfileobj(response.raw, temp_file)
                temp_file_name = temp_file.name
            ECLI_cache[cleaned_ecli] = temp_file_name
            try:
                with open(temp_file_name, 'rb') as file:
                    root = ET.parse(file).getroot()
                logging.info(f"Successfully fetched and parsed ECLI: {cleaned_ecli}")
            except Exception as e:
                logging.error(f"Error parsing fetched XML for {cleaned_ecli}: {e}")
                print(f"Skipping ECLI due to parse error: {cleaned_ecli}")
                return None, None, None
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching ECLI data for {cleaned_ecli}: {e}")
            print(f"Skipping ECLI due to fetch error: {cleaned_ecli}")
            return None, None, None

    identifier_link = None
    for identifier_tag in root.findall('.//rdf:Description/dcterms:identifier', namespaces):
        if identifier_tag.text and identifier_tag.text.startswith('http'):
            identifier_link = identifier_tag.text
            break

    date_link = None
    for date_tag in root.findall('.//rdf:Description/dcterms:issued', namespaces):
        if date_tag.text:
            try:
                date_link = datetime.strptime(date_tag.text, '%Y-%m-%d').date()
            except ValueError:
                logging.warning(f"Invalid date format for ECLI {cleaned_ecli}: {date_tag.text}")
                date_link = None

    return root, identifier_link, date_link

def remove_html_tags(text):
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def get_synonyms(word):
    return [word]
