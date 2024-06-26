# 音声合成と偽音声検出

### <b>背景</b>

クライアントはサイバーセキュリティ業界に携わるテクノロジー企業です。クライアントは最先端のテクノロジーを提供することにより、個人や組織が安全なデジタルプレゼンスを持つよう支援するシステムを構築することに焦点を当てています。また、オーディオとビデオのメディアが本物か偽物かを判明するため、データ駆動型テクノロジーを用いた製品とサービスを提供し、顧客のセキュリティを確保します。

このプロジェクトの目標は、音声（又は発話文）を特定の話者の声に変換することで音声の合成ができるシステム、さらに音声が本物か偽物かを検出するシステムを構築することです。

### <b>データ記述</b>

今回のプロジェクトで使用したデータセットは下記の2つです。どちらのデータセットも一般に公開されています。

<u>TIMITデータセット</u>:

TIMITの音声コーパスは、音響・音韻学研究や自動音声認識システムの開発と評価のための音声データを提供するよう作られています。データセットには、アメリカの8つの主要な方言地域からそれぞれ630人の話者が語った10文、合計6300の文の音声が含まれています。

データセットリンク: https://github.com/philipperemy/timit

<u>コモンボイス（Common Voice）データセット</u>:

コモンボイス（Common Voice）は、人が実際にどのように話すかを機械に教えることを支援するモジラ（Mozilla）の取り組みの一環であります。データセットは、コモンボイスウェブサイトのユーザーが読み取った音声データのコーパスであり、多数のパブリックドメインからのブログ投稿、古い書籍、映画、演説などのテキストデータに基づいています。。

データセットリンク: https://commonvoice.mozilla.org/en/datasets

### <b>目標</b>

上記で述べたようにこのプロジェクトの目標の1つは、音声が合成生成されたものかどうかを検出するための機械学習システムを構築することです。<br>
これを実現するには、以下のステップが必要になります。
1. ソース話者の音声（又は発話文）をターゲット話者の音声に変換する、音声クローニング・合成システムを構築
2. 音声が自然な話し言葉か、それとも機械によって合成生成されたものかを検出する機械学習システムを構築

音声合成システムの構築には、多様な話者の発話（すなわち音声）と発話文（すなわちテキスト）データが整列されているTIMITデータセットを使用します。

偽音声検出システムの構築には、数千個の自然な発話音声で構成されているコモンボイスデータセットを使用します。データセットが大きいため、サンプリングを通じてサブセットを使用するのも可能です。そのデータを正の例、すなわち人間による自然な音声として、音声合成システムによって生成された合成音声は負の例、すなわち機械による偽音声としてモデルトレーニングを行います。

### <b>成功指標</b>

<u>音声合成システム</u>

生成されたターゲット話者の音声のクォリティーを評価するため、単語誤り率（Word Error Rate、WER）と話者分類モデルの精度を成功指標として採用します。

<u>偽音声検出システム</u>

人間による音声の中から、音声合成システムによる偽音声を分別する性能の評価には、Fスコアを成功指標として採用します。

### <b>結果</b>

<u>音声合成システム</u>

1. 用いられたアルゴリズム／モデル

    17種類の異なるアルゴリズムをテストし、優れた性能を持つ以下の2つのアルゴリズムを特定しました。
    - vits: <a href='https://pypi.org/project/TTS/'>TTS</a>ライブラリーの音声変換モデル
    - speech_generator: <a href='https://pypi.org/project/Voice-Cloning/'>ボイスクローニング（Voice Cloning）</a>ライブラリーの音声合成・生成モデル

    アルゴリズムは、話者分類の精度に加え、以下の5つの評価指標を用いて評価されました。

    - 単語誤り率（Word Error Rate、WER）
    - 文字誤り率（Character Error Rate、CER）
    - 一致エラー率 (Match Error Rate、MER)
    - 単語情報損失率（Word Information Lost、WIL）
    - 単語情報保持率（Word Information Preserved、WIP）

2. 合成された音声のクォリティー

    - 原文（すなわち発話文）と合成音声を文字起こししたテキスト間の単語誤り率（WER）：<br>
    成功指標の1つである単語誤り率の算出には、音声合成アルゴリズムによって生成された音声の発話文が必要でした。<br>
    自動的な合成音声の文字起こしのため、<a href='https://pypi.org/project/SpeechRecognition/'>スピーチレコグニション（Speech Recognition）</a>ライブラリーを通じて、グーグル（Google）の音声認識APIを活用しました。<br><br>
    「vits」モデルによって生成された合成音声の平均単語誤り率は0.12、「speech_generator」の場合は平均単語誤り率が0.34でした。<br>
    「speech_generator」の単語誤り率は低い方ではありませんでしたが、話者分類の精度への好影響が判明され（下記に詳細あり）、採用することになりました。<br><br>
    単語誤り率以外、「vits」モデルの平均文字誤り率（CER）は0.04であり、「vits」モデルの優れた性能が判明しました。さらに「vits」モデルは、完璧な文字起こしができた合成音声（すなわち、オリジナル発話文と合成音声の発話文が完全に一致）もいくつか生成しました。
   
    - 話者分類モデルの精度：<br>
    本物の音声を用いてニューラルネットワークモデルを構築し、その話者分類モデルがどの程度の精度で合成音声の話者を分別できるのかを検証しました。精度が高いほど本物の音声と合成音声の間の差が少ないことになります。<br><br>
    検証の結果、分類モデルは0.83の精度で合成音声の話者を分別できました。各話者において利用できたデータはたった2-3秒程度の非常に短い音声データ10個だけだったのを考慮すると、分類モデルの精度は良好と見なされます。<br><br>
    また、「speech_generator」モデルで生成された合成音声が「vits」モデルより正確に分類されることが多かったです。これはターゲット話者の音声に、より類似している合成音声の生成が出来る「speech_generator」モデルの優れた性能を示唆します。

<u>偽音声検出システム</u>

1. 二値分類（本物の音声 vs. 合成音声）におけるF値

    偽音声検出のため、もう1つのニューラルネットワークモデルを構築しました。その二値分類モデルはテストデータで完璧なF値を達成し、最高の精度と再現率で偽音声を検出できる能力を証明しました。

    合成音声の品質は既に確認済みであることを考慮すると、本物の音声の中で合成音声を見つけられる偽音声検出システムの性能は非常に高いと考えられます。

<u>まとめ</u>

この度のプロジェクトを通じて得られた知識と分類モデルは、音声データが関わるさまざまなビジネス課題の解決に役立つ大きな潜在力／可能性を持っていると思います。今後のイテレーションでは、より多様な音声データを取り入れることでシステムの堅牢性と汎用性を向上させることができると考えられます。

### <b>ノートブック</b>

詳細については、<a href='https://github.com/henryhyunwookim/JAPANESE-VoiceCloningAndFakeAudioDetection/blob/main/VoiceCloningAndFakeAudioDetection.ipynb'>このノートブック</a> を直接ご参照ください。

ノートブック（VoiceCloningAndFakeAudioDetection.ipynb）をローカル環境で実行するには、このリポジトリをクローン（複製）またはフォーク（派生）し、下記のコマンドを実行することで、必要なライブラリーをインストールしてください。

pip install -r requirements.txt

##### <i>* アプジバ（Apziva）に関連</i>