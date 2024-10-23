from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

app = Flask(__name__)

# 設定上傳檔案的目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # 讀取 CSV 資料
        data = pd.read_csv(file_path)

        # 資料格式處理
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data['y'] = data['y'].str.replace(',', '').astype(float)

        # 準備數據格式給 Prophet
        df = data.rename(columns={'Date': 'ds', 'y': 'y'})

        # 使用 Prophet 模型預測
        model = Prophet(changepoint_prior_scale=0.5, yearly_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df)

        # 預測未來 60 天
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        # 畫出預測結果圖
        plt.figure(figsize=(14, 7))
        plt.plot(df['ds'], df['y'], color='black', label='Actual Data')
        plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Predicted Data')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)
        historical_average = df['y'].mean()
        plt.axhline(historical_average, color='blue', linestyle='--', label='Historical Average')
        plt.title('Stock Price Prediction with Prophet')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid()

        # 儲存圖表
        image_path = 'static/prediction.png'
        plt.savefig(image_path)
        plt.close()

        return render_template('index.html', image_path=image_path)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
