const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');
const fs = require('fs');
const natural = require('natural');

const csvFilePath = './files/sms_spam_dataset.csv';

const textData = [];
const labelData = [];

fs.createReadStream(csvFilePath)
  .pipe(csv({skipLines: 1})) //skip header line
  .on('data', (row) => {
    textData.push(row.sms);
    labelData.push(row.label);
  })
  .on('end', () => {
    console.log('CSV file successfully processed');

    // label numeric vals: 0 = ham, 1 = spam
    const numericLabelData = labelData.map((label) => parseInt(label));

    // arrange text data
    

    // normalise
    const normalizedTextData = dataset.map((text) => text.div(tf.scalar(255)));
    const labelTensor = tf.tensor1d(numericLabelData);

    // create and compile
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // train
    normalizedTextData
      .forEachAsync((text) => {
        const textTensor = tf.tensor2d([text.dataSync()], [1, 1]);
        return model.fit(textTensor, labelTensor, {
          epochs: 10,
          batchSize: 32,
          callbacks: tf.node.tensorBoard('logs'),
        });
      })
      .then(() => {
        const prediction = model.predict(normalizedTextData);

        console.log('Prediction:', prediction);
      });
  });