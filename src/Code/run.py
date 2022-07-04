import tamil
from PyPDF2 import PdfFileReader
from flask import Flask, flash, render_template, request, send_file
from flask_bootstrap import Bootstrap
import os
from src.Code.QAG_Module import processFile

app= Flask(__name__)
Bootstrap(app)

app.config["FILE_UPLOADS"] = "C:\\Users\ASUS\Desktop\Research Work Upload documents\Github-Aaivu\Question-Answer-generation-system-in-Tamil-language\src\Code\\files\\uploaded_files"
app.config["TXT_CONVERSION_FILE_PATH"] = "C:\\Users\ASUS\Desktop\Research Work Upload documents\Github-Aaivu\Question-Answer-generation-system-in-Tamil-language\src\Code\\files\pdftotxtconversion.txt"
app.config["ALLOWED_FILE_EXTENSIONS"] = ["pdf", "doc", "txt", "docx"]
app.config['SECRET_KEY'] = 'thisisasecret'
app.config['QUESTION_ANSWER_FILE_PATH'] = "C:\\Users\ASUS\Desktop\Research Work Upload documents\Github-Aaivu\Question-Answer-generation-system-in-Tamil-language\src\Code\\files\questionanswers.txt"
app.config["MAX_IMAGE_FILE_SIZE"] = 10 * 1024 * 1024


def allowed_files(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]
    if ext.lower() in app.config["ALLOWED_FILE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_file_size(file_size):
    if int(file_size) <= app.config["MAX_IMAGE_FILE_SIZE"]:
        return True
    else:
        return False


def read_pdf_file(file):
    pdf = PdfFileReader(file)
    with open(app.config['TXT_CONVERSION_FILE_PATH'], 'w', encoding="utf-8") as f:
        for page_num in range(pdf.numPages):
            # print('Page: {0}'.format(page_num))
            page_obj = pdf.getPage(page_num)

            try:
                txt = page_obj.extractText()
                print(''.center(100, '-'))
            except:
                pass
            else:
                f.write('Page {0}\n'.format(page_num + 1))
                f.write(''.center(100, '-'))
                f.write(txt)
        f.close()


def read_txt_file(filename):
    with open(os.path.join(app.config["FILE_UPLOADS"], filename), encoding="utf-8") as f:
        file_content = f.read()
        flash('File uploaded successfully. Please review content before generating Q&A.', 'success')
        return file_content


def write_qa_file(writefile, question, answer):
    question = question.strip('..')
    writefile.write(question + "?\n")
    print(answer)
    writefile.write(answer + "\r\n")
    return True


def write_gap_fill_question(writefile, question, answer):
    writefile.write(question + "\n")
    print(answer)
    writefile.write(answer + "\r\n")
    return True


@app.route("/download")
def download_txt_file():
    return send_file(app.config["QUESTION_ANSWER_FILE_PATH"], as_attachment=True, cache_timeout=0)


@app.route("/", methods=["GET", "POST"])
def view_qa_generation_page():
    file_content = ""
    generated_qa = ""
    if request.method == "POST":
        if request.form['submitBtn'] == 'uploadFile':
            if request.files:
                file = request.files["upload-file"]
                if allowed_files(file.filename):
                    ext = file.filename.rsplit(".", 1)[1].lower()
                    file.save(os.path.join(app.config["FILE_UPLOADS"], file.filename))
                    if ext == 'pdf':
                        read_pdf_file(file)
                        file_content = read_txt_file('pdftotxtconversion.txt')
                        file_content = tamil.txt2unicode.auto2unicode(file_content)
                    else:
                        file_content = read_txt_file(file.filename)

                else:
                    flash('The file uploaded is not supported by the application.', 'danger')

        elif request.form['submitBtn'] == 'generateQA':
            file_content = request.form['textContent']
            processFile.process_file(file_content, app.config['QUESTION_ANSWER_FILE_PATH'])
            with open(app.config["QUESTION_ANSWER_FILE_PATH"], encoding="utf-8") as f:
                generated_qa = f.read()

    return render_template("questionAnswerGenerator.html", content=file_content, questionAnswers=generated_qa)


@app.route("/named-entity-recognition")
def view_ner_page():
    return render_template("ner.html", title="Named Entity Recognition")


@app.route("/manual-tagging")
def view_manual_tagging_page():
    return render_template("manualTagging.html", title="Manual Tagging")


@app.route("/gazeteers")
def view_gazetteers_page():
    return render_template("gazetteers.html", title="Gazetteers")


if __name__ =='__main__':
    app.run(debug=True)