- Các hàm các phương thức xử lý gán nhãn ngữ nghĩa.
- Xử lý cho tiếng việt: http://viet.jnlp.org/kien-thuc-co-ban-ve-xu-ly-ngon-ngu-tu-nhien/xu-ly-tieng-viet-bang-python-1

GÁN NHÃN NGỮ NGHĨA CHO TỪ TIẾNG ANH TRONG SONG NGỮ ANH VIỆT

GÁN NHÃN NGỮ NGHĨA BẰNG PHƯƠNG PHÁP NEURON NETWORK


Background Neuron NetWork: http://bis.net.vn/forums/p/482/1122.aspx
Mô Hình

    Bi-Sentence =>Word Embedding => Bi-LSTM => XOR thành 1 vector => LSTM => Softmax => Câu đã gán nhãn
- Bi-Sentence:
    - Các cặp câu song ngữ (Câu tiếng Anh nằm trong bộ Semcor)
- Word Embedding:
    -Yêu cầu:
        - Bộ data song ngữ Anh-việt dùng làm dữ liệu train cho embedding (có thể lấy từ wiki...)
        - Tool tách từ cho tiếng Việt (tiếng Anh không cần).
        - Source code word Embedding tham khảo tại đây: http://blog.duyet.net/2017/04/nlp-truyen-kieu-word2vec.html#.WdWdQ8JSDIU
        - Dữ liệu test các cặp câu song ngữ Anh-Việt (Các cặp song ngữ được dịch chính xác từ semcor).
    - Thực hiện
        - Word Embedding trên từng ngôn ngữ riêng biệt.
        - Tiếng Anh thì train bộ ngữ liệu tiếng Anh.
        - Tiếng Việt train trên bộ ngữ liệu tiếng Việt.
        - Sử dụng thư viện gensim để w2v (trên từ từ):
        - tạo 1 batch trên toàn bộ
        - Kết quả:
            Câu tiếng Anh: Ma trận N x M (M từ trong câu và N chiều)
            Câu tiếng Việc: Ma trận N x M (M từ trong câu và N chiều)
            Kết hợp: như thế nào là tốt nhất ???
    - Kiểm tra:
        - Đo độ tương đồng của 2 câu song ngữ với nhau
- LSTM
    - Yêu cầu:
        - Các bộ tham số ban đầu: W, U, bias ...
        - Đầu vào là các vector: tính ở word embedding
        - các one-hot vector: chiều dài là số lương sense của mỗi từ
    - Thực hiện:
        - Tối ưu trọng số W và U
        - Trainning:
            - Input đầu vào (1 câu là một ma trận) gồm:
                - Mỗi từ là một vector
                - Mỗi từ có số phân lớp (số lượng sense) và 1 ground-truth => one-hot vector: nhãn đúng của từ đó có giá trị là 1, còn lại là 0
                - Độ dài chuỗi:
                - leaning rate: tỉ lệ học
            - Kết quả: các bộ tham số được học
        - Testing:
            - Input đầu vào (1 câu là 1 ma trận) gồm:
                - Mỗi từ là một vector.
                - Số phân lớp của từng từ (số lượng sense)
            - Kết quả
                - Tại mỗi từ xác định được nó thuộc phân lớp nào.
                - Câu nào được gán nhãn xong sẽ được đưa vào dữ liệu train và train lại bộ trong số.

- Softmax
    - Input: tại mỗi từ, input là vector có số chiều là số phân lớp (số lượng sense của từ đó)
    - Tính toán: {k}
- Trên các bộ train:
    - Tại mỗi nhãn, lấy các từ đã gán tìm trong bộ từ điển wordnet

Một số hạn chế:
    - Thực hiện trên từ đơn
    - Gạp cụm từ không thể gán nhãn được
    - Một số nhãn bị sai (ví dụ: từ br-a01-2 overall%5:00:00:gross:00   => đúng là overall%3:00:00:general:00)
    - Tại data train, lấy các nhãn để search
    - Gặp mấy trường hợp này thì chấp nhận sai số
    - Các từ được gán nhãn nhưng tìm trong wordnet thì không thấy: br-a01-4    28  32  such%5:00:01:specified:00   such
    - Một số từ có dấu '-' (từ ghép trong tiếng anh) có một số từ tìm được trong wordnet (ví dụ: hard-fought), một số khác lại ko tìm thấy (bỏ dấu gạch nối thì tìm được- ví dụ từ over-all)
    - Trong bộ wordnet được định nghĩa tới 5 từ loại chính ()
    - Một số câu có mạo từ "a, an, the" tìm có trong wordnet, chắc phải loại bỏ các từ này khi tìm trong wordnet
Một số source tham khảo:
https://github.com/nicodjimenez/lstm/blob/master/test.py
https://github.com/jiexunsee/Numpy-LSTM/blob/master/numpylstm.py
https://gist.github.com/stober/1946926
https://github.com/statguy/wsd
# Update trọng số: https://machinelearningcoban.com/2017/01/16/gradientdescent2/
#Cách tính LSMT: http://practicalcryptography.com/miscellaneous/machine-learning/graphically-determining-backpropagation-equations/
#hàm cost: https://stackoverflow.com/questions/36355891/different-loss-functions-for-backpropagation

error = x - y (x: kết quả tính toán, y: kết quả thật sự)
output = x * (1-x) (x: kết quả tính toán)
delta_out = error * output


LSTM nhiều lớp: https://mazsola.iit.uni-miskolc.hu/~czap/letoltes/IS14/IS2014/PDF/AUTHOR/IS140552.PDF
Xây dựng LSTM nhiều lớp:
Forward:
    Input: Vector M chiều (M: không gian input, N: số step)
    LSTM đầu tiên:
        input: M chiều
            forward step: chuyển đổi từ không gian M chiều -> K chiều (số chiều lớp ẩn) (Cần vector trọng số )
            backward step: Chuyển đổi không M chiều -> K chiều
        Lớp ẩn: (M -> K chiều
        output: L chiều (chuyển đổi từ K chiều -> L chiều)
            Kết hợp 2 bước này -> vector output L chiều
            Lưu Ý: tại mỗi step số chiều L là khác nhau (ví dụ: mỗi từ có số phân lớp khác nhau nên số lớp cũng khác nhau)
    LSTM tiếp theo:
        input L chiều
        Lớp ẩn: Chuyển đổi L sang K chiều (Cần 1 vector trọng số W (LxM chiều)-> L tại mỗi bước khác nhau, hông lẽ cần n vector L xM chiều sao?)
        output: chuyển đổi K chiều sang L chiều (dùng cùng 1 vector)

    Lúc backpropagation khá cực đây.
    Đây mới là trên đơn ngữ

KHởi tạo trong số: "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"
    Các tham số W, U, V: thuộc khoản: [-sqrt(6/(r+c)); -sqrt(6/(r+c))]
    Bias: khởi tạo các vector 0, riêng công forget khởi tạo vector 1
