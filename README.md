# Ham or Spam?

I like ham, I definitely do not like spam. So I trained an AI to help me categorise my SMS, learning from a CSV dataset that outputs 0 = ham, 1 = spam.

### Tech

- Common JavaScript
- Node.js
- TensorFlow.js
- csv-parser library
- NPM for package management

### Run/Demo locally

1. Fork repository
2. Make sure you have Python and Node v17.9.1 (version matters) installed and run `npm i`
3. Run `node tf-model.js`

### To-Do

- ~~Important: Fix bug where predictions all say it's ham even when label is spam, making me eat the yucky spam~~
- Build a frontend with a text input to provide an SMS and have the model categorise it
- Refactor from commonjs to module es6
- Put that frontend and our Node.js server live and help others avoid eating spam
  <br></br>

---

### Biggest challenges faced when building "Ham or Spam"

- Understanding the shape of the dataset to properly have my CSV parser ingest it
- Had to downgrade my Node.js version to workaround a dependency issue with TensorFlow for node :(
- Working around tokenisation to use csv data in the tensor
- Overall understanding ML as a first project

### Conclusion

TensorFlow.js is a great start for a machine learning beginner like me who doesn't want to learn python at the moment. The official documentation is clear and there's many resources to troubleshoot issues. I'd recommend using a simple dataset first to not overcomplicate data ingestion.
<br><br/>

---

<sub> made with ♡ by eni </sub>
