#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Intel社のサンプルを元にy.fukuharaが簡略化と日本語コメントの追記（2020/06/14）

"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore

# コマンド実行時の引数を読む
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-f", "--face", help="Required. Path to an .xml file with a trained model.", required=True,  type=str)
    args.add_argument("-a", "--age", help="Required. Path to an .xml file with a trained model.", required=True,  type=str)

    args.add_argument("-e", "--encoder", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-d", "--decoder", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-l", "--labels", help="Optional. Path to a labels mapping file", default="./MLdesign/kinetics_400_labels.csv", type=str)


    return parser


# 実行プログラムのメイン部分
def main():
    # ログ出力の設定
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    # コマンドライン引数を読み込む
    args = build_argparser().parse_args()


    # 初期化
    log.info("推論エンジンの設定")
    ie = IECore()

    # Faceモデルの読み込み
    model_xml_1 = args.face
    model_bin_1 = os.path.splitext(model_xml_1)[0] + ".bin"
    log.info("Faceモデルファイルを確認:\n\t{}\n\t{}".format(model_xml_1, model_bin_1))
    if os.name == 'posix':
        net_1 = ie.read_network(model=model_xml_1, weights=model_bin_1)
    else:
        net_1 = IENetwork(model=model_xml_1, weights=model_bin_1)
    exec_net_1 = ie.load_network(network=net_1, device_name="CPU")

    # Age genderモデルの読み込み
    model_xml_2 = args.age
    model_bin_2 = os.path.splitext(model_xml_2)[0] + ".bin"
    log.info("Ageモデルファイルを確認:\n\t{}\n\t{}".format(model_xml_2, model_bin_2))
    if os.name == 'posix':
        net_2 = ie.read_network(model=model_xml_2, weights=model_bin_2)
    else:
        net_2 = IENetwork(model=model_xml_2, weights=model_bin_2)
    exec_net_2 = ie.load_network(network=net_2, device_name="CPU")
    
    
    # encoderモデルの読み込み
    model_encoder_xml = args.encoder
    model_encoder_bin = os.path.splitext(model_encoder_xml)[0] + ".bin"
    log.info("エンコーダーモデルファイルをロード:\n\t{}\n\t{}".format(model_encoder_xml, model_encoder_bin))
    #net_encoder = ie.read_network(model=model_encoder_xml, weights=model_encoder_bin)
    if os.name == 'posix':
        net_encoder = ie.read_network(model=model_encoder_xml, weights=model_encoder_bin)
    else:
        net_encoder = IENetwork(model=model_encoder_xml, weights=model_encoder_bin)
    exec_net_encoder = ie.load_network(network=net_encoder, device_name="CPU")


    # decoderモデルの読み込み
    model_decoder_xml = args.decoder
    model_decoder_bin = os.path.splitext(model_decoder_xml)[0] + ".bin"
    log.info("デコーダーモデルファイルをロード:\n\t{}\n\t{}".format(model_decoder_xml, model_decoder_bin))
    #net_decoder = ie.read_network(model=model_decoder_xml, weights=model_decoder_bin)
    if os.name == 'posix':
        net_decoder = ie.read_network(model=model_decoder_xml, weights=model_decoder_bin)
    else:
        net_decoder = IENetwork(model=model_decoder_xml, weights=model_decoder_bin)
    exec_net_decoder = ie.load_network(network=net_decoder, device_name="CPU")
    
    
#    # 顔モデルのネットワークに関するデータを取得
    log.info("入出力層の情報を取得")
    input_blob_1 = next(iter(net_1.input_info))
    out_blob_1 = next(iter(net_1.outputs))
    net_1.batch_size = 1

    # 年齢性別モデルのネットワークに関するデータを取得
    input_blob_2 = next(iter(net_2.input_info))
    out_2 = iter(net_2.outputs)
    out_blob_2_age = next(out_2) #年齢
    out_blob_2_gender = next(out_2)#性別
    net_2.batch_size = 1
    
    



    # encoderモデルのネットワークに関するデータを取得
    log.info("入出力層の情報を取得")
    input_blob_encoder = next(iter(net_encoder.input_info))
    out_encoder = iter(net_encoder.outputs)
    out_blob_encoder = next(iter(out_encoder))
    net_encoder.batch_size = 1

    # decoderモデルのネットワークに関するデータを取得
    input_blob_decoder = next(iter(net_decoder.input_info))
    out_decoder = iter(net_decoder.outputs)
    out_blob_decoder = next(iter(out_decoder))
    net_decoder.batch_size = 1
    
    
    
    
    

#    # 入力層への入力形式を取得する
    n1, c1, h1, w1 = net_1.inputs[input_blob_1].shape
    log.info(str(net_1.inputs[input_blob_1].shape))
    log.info(str(net_1.outputs[out_blob_1].shape))

    n2, c2, h2, w2 = net_2.inputs[input_blob_2].shape
    log.info(str(net_2.inputs[input_blob_2].shape))
    log.info(str(net_2.outputs[out_blob_2_age].shape))
    log.info(str(net_2.outputs[out_blob_2_gender].shape))

#    [ INFO ] [1, 3, 300, 300]
#    [ INFO ] [1, 1, 200, 7]
#    [ INFO ] [1, 3, 62, 62]
#    [ INFO ] [1, 1, 1, 1]
#    [ INFO ] [1, 2, 1, 1]


    
    # encoderモデルの入出力層の形式を取得する
    n, c, h, w = net_encoder.inputs[input_blob_encoder].shape
    log.info(str(net_encoder.inputs[input_blob_encoder].shape))
    log.info(str(net_encoder.outputs[out_blob_encoder].shape))

    # decoderモデルの入出力層の形式を取得する
    b, t, e = net_decoder.inputs[input_blob_decoder].shape
    log.info(str(net_decoder.inputs[input_blob_decoder].shape))
    log.info(str(net_decoder.outputs[out_blob_decoder].shape))

    
#    [ INFO ] [1, 3, 224, 224]
#    [ INFO ] [1, 512, 1, 1]
#    [ INFO ] [1, 16, 512]
#    [ INFO ] [1, 400]
    
    
    
    # ラベルファイルがあれば読み込む
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=',', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    
    

    # カメラをオープン
    cap = cv2.VideoCapture(0)

    i = 0
    embedding = np.empty((1,16,512))

    while True:
        # VideoCaptureから1フレーム読み込む
        ret, image = cap.read()
        org_image = image
        org_image_2 = image
        image_2 = image

        ih, iw = image.shape[:-1]
        # 画像を入力層に合わせてリサイズする
        if (ih, iw) != (h1, w1):
            image = cv2.resize(image, (w1, h1))
            
        # 画像データ形式をモデルに合わせて変更（H,W,CからC,H,Wの並び順にする）
        image = image.transpose((2, 0, 1))



        # 推論実行
        res = exec_net_1.infer(inputs={input_blob_1: image})






        # 結果の出力
        res = res[out_blob_1] # 顔検知の出力
        objects = {}
        data = res[0][0]
        for number, proposal in enumerate(data):
            if proposal[2] > 0:
                imid = np.int(proposal[0])
                # ラベルIDの取得
                label = np.int(proposal[1])
                # 確信度
                confidence = proposal[2]
                # 座標の取得(0-1の範囲なので、縦横ピクセル倍する)
                xmin = np.int(iw * proposal[3])
                ymin = np.int(ih * proposal[4])
                xmax = np.int(iw * proposal[5])
                ymax = np.int(ih * proposal[6])
                # 信頼度が0.5以上なら出力する
                if proposal[2] > 0.5:
                    try:
                        det_label =  "{}".format(label)
                        if not imid in objects.keys():
                            objects[imid] = []

                        # 元画像から顔部分を切り出して、新しい画像を作る
                        face_image = org_image[ymin:xmin, ymax:xmax]

                        # 顔画像をage_genderモデルの入力層に合わせてリサイズする
                        fh, fw = face_image.shape[:-1]
                        if (fh, fw) != (h2, w2):
                            face_image = cv2.resize(face_image, (w2, h2))
                        # 画像データ形式をモデルに合わせて変更（H,W,CからC,H,Wの並び順にする）
                        face_image = face_image.transpose((2, 0, 1))

                        # 年齢・性別の推論実行
                        age_gender = exec_net_2.infer(inputs={input_blob_2: face_image})

                        # 年齢を取得
                        age = age_gender[out_blob_2_age]
                        # 性別を取得
                        gender_arr = np.squeeze(age_gender[out_blob_2_gender])
                        if gender_arr[0] >= gender_arr[1]:
                            gender = "Female"
                        else:
                            gender ="Male"
                        # 各情報を配列に格納
                        objects[imid].append([xmin, ymin, xmax, ymax, det_label, int(age*100), gender])
                    #　try以降で問題があった場合、表示する。
                    except Exception as e:
                        print("error:", e.args)
                        continue


        # 枠と年齢・性別の描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(objects)>0:
            for box in objects[0]:
                if box[6] == "Male":
                    # BGR
                    color = (255,0,0)
                else:
                    # BGR
                    color = (0,0,255)
                    
                cv2.rectangle(org_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(org_image, "Age: "+ str(box[5]) + "  Gender: "+ box[6] ,(box[0],box[1]-5), font, 1,color,2,cv2.LINE_AA)
                
        cv2.imshow('result', org_image)





        # 画像を入力層に合わせてリサイズする
        if (ih, iw) != (h, w):
            image_2 = cv2.resize(image_2, (w, h))
        # 画像データ形式をモデルに合わせて変更（H,W,CからC,H,Wの並び順にする）
        image_2 = image_2.transpose((2, 0, 1))

        # 結果処理
        # エンコードの推論実行
        encode = exec_net_encoder.infer(inputs={input_blob_encoder: image_2})


        # エンコードの出力データの読み込み。
        encode = encode.get(out_blob_encoder)
        # 16フレーム分ためる
        if i<16:
            embedding[0][i] = np.squeeze(encode)
        else:
            embedding[0][15] = np.squeeze(encode)
            
        i=i+1
        if i>=16:
            # 16フレーム分のエンコーダーからの出力をデコーダーに投入する
            decode = exec_net_decoder.infer(inputs={input_blob_decoder: embedding})
            decode = decode.get(out_blob_decoder)
            # デコーダーの結果から、最大値のインデックスを取得
            idx = np.argmax(np.squeeze(decode))
            # ラベルを取得
            det_label = labels_map[idx] if labels_map else "{}".format(idx)
            #print(det_label)

            # 16フレームを超えたら、ローテーションする
            embedding = np.roll(embedding, -1, axis=1)


            out_act_ls = ["playing saxophone", "playing trombone","beatboxing","unboxing","cleaning shoes","reading book","reading newspaper","playing accordion"]
            color = (200,100,60)
            for out_act in out_act_ls:
                # 不正行為
                if det_label == out_act:
                    # ラベルの描画
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(org_image, det_label,(50,50), font, 1,color,2,cv2.LINE_AA)
                    cv2.putText(org_image, "cheating!",(950,50), font, 2.0,(0,0,255),5,cv2.LINE_AA)
                # 不正行為ではない
                else:
                    # ラベルの描画
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(org_image, det_label,(50,50), font, 1,color,2,cv2.LINE_AA)


        #画像の表示
        cv2.imshow('result', org_image)






        # ESCキーで終了
        k = cv2.waitKey(1)
        if k == 27:
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)

