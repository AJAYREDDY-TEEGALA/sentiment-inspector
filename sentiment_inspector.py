# sentiment_inspector.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import io

# Load NLP models
vader_analyzer = SentimentIntensityAnalyzer()
hf_analyzer = pipeline("sentiment-analysis")

# Main App Class
class SentimentInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Inspector")
        self.root.geometry("800x600")

        self.model_choice = tk.StringVar(value="VADER")

        self.build_ui()

    def build_ui(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        # Input Section
        tk.Label(frame, text="Enter Text:").grid(row=0, column=0)
        self.text_input = tk.Text(frame, height=5, width=60)
        self.text_input.grid(row=1, column=0, columnspan=4)

        tk.Button(frame, text="Analyze Text", command=self.analyze_text).grid(row=2, column=0, pady=5)
        tk.Button(frame, text="Upload CSV", command=self.load_csv).grid(row=2, column=1, pady=5)
        tk.Button(frame, text="Fetch from API", command=self.fetch_api_data).grid(row=2, column=2, pady=5)
        tk.Button(frame, text="Export Results", command=self.export_results).grid(row=2, column=3, pady=5)

        # Model Choice
        tk.Label(frame, text="Choose Model:").grid(row=3, column=0)
        tk.Radiobutton(frame, text="VADER", variable=self.model_choice, value="VADER").grid(row=3, column=1)
        tk.Radiobutton(frame, text="HuggingFace", variable=self.model_choice, value="HF").grid(row=3, column=2)

        # Treeview Table
        self.tree = ttk.Treeview(self.root, columns=("Text", "Sentiment", "Score"), show="headings")
        self.tree.heading("Text", text="Text")
        self.tree.heading("Sentiment", text="Sentiment")
        self.tree.heading("Score", text="Score")
        self.tree.pack(pady=10, fill=tk.BOTH, expand=True)

        # Chart Area
        self.fig, self.ax = plt.subplots(figsize=(5, 2))
        self.chart = FigureCanvasTkAgg(self.fig, master=self.root)
        self.chart.get_tk_widget().pack(fill=tk.BOTH, expand=False)

        self.results = []

    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            self.process_texts([text])

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            column = messagebox.askquestion("Column Selection", f"Using first text column: {df.columns[0]}. Continue?")
            if column == "yes":
                self.process_texts(df[df.columns[0]].astype(str).tolist())

    def fetch_api_data(self):
        try:
            url = "https://newsapi.org/v2/everything?q=apple&from=2025-06-17&to=2025-06-17&sortBy=popularity&apiKey=547ed6b2ef9b4be3abbedffbdca3dbec"
            response = requests.get(url)
            data = response.json()
            texts = [item["content"] for item in data["articles"]]
            print(texts)
            self.process_texts(texts)
        except Exception as e:
            messagebox.showerror("API Error", str(e))

    def process_texts(self, texts):
        self.results.clear()
        self.tree.delete(*self.tree.get_children())
        self.ax.clear()

        def worker():
            sentiments = []
            for text in texts:
                if self.model_choice.get() == "VADER":
                    score = vader_analyzer.polarity_scores(text)["compound"]
                    sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                else:
                    result = hf_analyzer(text[:512])[0]  # Truncate for large text
                    sentiment = result['label']
                    score = result['score']

                self.results.append((text, sentiment, round(score, 3)))
                self.tree.insert("", tk.END, values=(text[:50] + ("..." if len(text) > 50 else ""), sentiment, round(score, 3)))
                sentiments.append(sentiment)

            self.plot_results(sentiments)

        threading.Thread(target=worker).start()

    def plot_results(self, sentiments):
        from collections import Counter
        counts = Counter(sentiments)
        self.ax.bar(counts.keys(), counts.values(), color=['green', 'blue', 'red'])
        self.ax.set_title("Sentiment Distribution")
        self.chart.draw()

    def export_results(self):
        if not self.results:
            messagebox.showwarning("No Data", "No results to export.")
            return
        df = pd.DataFrame(self.results, columns=["Text", "Sentiment", "Score"])
        file_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Results saved to {file_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SentimentInspectorApp(root)
    root.mainloop()
