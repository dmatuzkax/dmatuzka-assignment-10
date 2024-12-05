from flask import Flask, render_template, request, url_for
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer, get_tokenizer
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def index():
    # Get the data and create a model
    df = pd.read_pickle('image_embeddings.pickle')
    model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')

    # Find best matches among images using cosine similarity given query embedding
    def query(query_embedding):  
        query_embedding = query_embedding.squeeze(0).detach().cpu().numpy()
        embeddings = np.stack(df['embedding'].values)
        
        cosine_similarities = np.dot(embeddings, query_embedding)

        top_indices = np.argsort(cosine_similarities)[-5:][::-1]
        top_images = []
        for idx in top_indices:
            impath = '../static/coco_images_resized/' + df.iloc[idx]['file_name']
            similarity = cosine_similarities[idx]
            top_images.append((impath, similarity))
        
        return top_images
    
    # Save input image
    def save_image(image):
        filename = image.filename
        filepath = os.path.join('./static/images', filename)
        image.save(filepath)

        return filepath
    
    # Load images function for PCA
    def load_images(image_dir, max_images=None, target_size=(224, 224)):
        images = []
        image_names = []
        for i, filename in enumerate(os.listdir(image_dir)):
            if filename.endswith('.jpg'):
                img = Image.open(os.path.join(image_dir, filename))
                img = img.convert('L')  # Convert to grayscale ('L' mode)
                img = img.resize(target_size)  # Resize to target size
                img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
                images.append(img_array.flatten())  # Flatten to 1D
                image_names.append(filename)
            if max_images and i + 1 >= max_images:
                break
        return np.array(images), image_names
    
    # Transform for PCA image given by the user in a query
    def transform_image(filepath):
        images = []
        img = Image.open(filepath)
        img = img.convert('L')  # Convert to grayscale ('L' mode)
        img = img.resize((224, 224))  # Resize to target size
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        return img_array.flatten()

    # Logic for query  
    if request.method == "POST":
        # Check for the type of query
        query_type = request.form["type"]

        # Image Query
        if query_type == 'image':
            image = request.files['image']
            image_type = request.form['image-type']
            # Check whether to use CLIP or PCA embeddings
            if image_type == 'PCA':
                train_images, train_image_names = load_images('./static/coco_images_resized', max_images=2000, target_size=(224, 224))
                k = int(request.form['k'])
                pca = PCA(n_components=k)
                pca.fit(train_images)

                transform_images, transform_image_names = load_images('./static/coco_images_resized', max_images=10000, target_size=(224, 224))
                reduced_embeddings = pca.transform(transform_images)
                query_embedding = pca.transform([transform_image(save_image(image))])

                cosine_similarities = cosine_similarity(reduced_embeddings, query_embedding)
                cosine_similarities = cosine_similarities.flatten()
                top_indices = np.argsort(cosine_similarities)[-5:][::-1]

                top_images = [('../static/coco_images_resized/' + transform_image_names[idx], cosine_similarities[idx]) for idx in top_indices]
            else:
                filepath = save_image(image)
                
                image = preprocess(Image.open(filepath)).unsqueeze(0)
                query_embedding = F.normalize(model.encode_image(image))
                top_images = query(query_embedding)

        # Text Query
        if query_type == 'text':
            text = request.form['text']
            tokenizer = get_tokenizer('ViT-B-32')
            model.eval()
            text = tokenizer([text])

            query_embedding = F.normalize(model.encode_text(text))
            top_images = query(query_embedding)
        
        # Hybrid Query
        if query_type == 'hybrid':
            image = request.files['image']
            filename = image.filename
            filepath = os.path.join('./static/images', filename)
            image.save(filepath)
            text = request.form['text']

            image = preprocess(Image.open(filepath)).unsqueeze(0)
            image_query = F.normalize(model.encode_image(image))
            tokenizer = get_tokenizer('ViT-B-32')
            text = tokenizer([text])
            text_query = F.normalize(model.encode_text(text))

            lam  = float(request.form['weight'])

            query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query)
            top_images = query(query_embedding)
         

        return render_template("index.html", top_images=top_images)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)