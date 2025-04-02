const express = require('express');
const cors = require('cors');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;
const upload = multer({ dest: 'uploads/' });

// 미들웨어 설정
app.use(cors());
app.use(express.json());

// 모델 로드
async function loadModel() {
  model = await tf.loadLayersModel(`file://${__dirname}/model/model.json`);
  console.log('모델이 로드되었습니다.');
}

// 서버 시작 시 모델 로드
loadModel().catch(err => {
  console.error('모델 로드 실패:', err);
});

// 이미지 예측 API 엔드포인트
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: '이미지가 제공되지 않았습니다.' });
    }

    if (!model) {
      return res.status(500).json({ error: '모델이 아직 로드되지 않았습니다.' });
    }

    // 이미지 파일 읽기
    const imagePath = path.join(__dirname, req.file.path);
    const imageBuffer = fs.readFileSync(imagePath);
    
    // 이미지를 텐서로 변환
    const tfimage = tf.node.decodeImage(imageBuffer);
    const resized = tf.image.resizeBilinear(tfimage, [224, 224]); // 모델에 맞는 크기로 조정
    const normalized = resized.div(255.0).expandDims();
    
    // 예측 수행
    const predictions = await model.predict(normalized).data();
    
    // 파일 삭제
    fs.unlinkSync(imagePath);
    
    // 결과 포맷팅 (클래스별 확률)
    const classes = ['class1', 'class2', 'class3']; // 모델의 클래스명으로 변경하세요
    const result = Array.from(predictions).map((prob, i) => ({
      class: classes[i],
      probability: prob
    }));

    res.json({
      success: true,
      predictions: result
    });
    
  } catch (error) {
    console.error('예측 중 오류 발생:', error);
    res.status(500).json({ error: '예측을 처리하는 중 오류가 발생했습니다.' });
  }
});

// 서버 상태 확인용 엔드포인트
app.get('/status', (req, res) => {
  res.json({ status: 'online' });
});

app.listen(port, () => {
  console.log(`서버가 포트 ${port}에서 실행 중입니다.`);
});
