# Introduction 

## What is '리더스' 

'리더스' 어플은 사용자가 읽은 책을 기록하고, 마음에 드는 구절을 강조하며 감상평을 남길 수 있는 독서 플랫폼이다. 기록을 통해 사용자는 자신의 독서 이력을 관리 및 공유하며, '북클럽'이라는 기능을 활용해 '리더스'가 선정한 다양한 주제의 책을 다른 어플 사용자들과 함께 읽는다. 


## Purpose of the project 

1. 사용자의 책 스크랩 문구를 분석하는 텍스트 마이닝 

2. 개인화된 책을 추천하는 알고리즘 개발 

추천 알고리즘을 통해 북클럽에서 사용자의 취향과 기준에 따른 최적의 책을 추천받으며, 스크랩 문구 텍스트 마이닝을 연령에 따라 분류하여 가장 높은 빈도수를 가진 단어를 각 연령 그룹의 대표 키워드로 설정해 개인화된 독서경험을 제공한다. 

---

# 1. tf-idf 기반 도서 및 문구 추천 알고리즘 

## Purpose 

이용자가 자신이 읽은 책에서 인상 깊었던 부분에 대한 스크랩 문구(구절)이 유저들 간 교류를 위한 합리적인 수단이라 판단했으며, 비슷한 특성의 유저 집단을 대표할 수 있는 키워드를 선정 후 키워드 관련 도서와 문구를 추천해준다면 유전 간 교유를 활발히 하고 '리더스'만의 독서 SNS라는 강점을 확고히 할 것이다. 




## Processing  

tf-idf 알고리즘을 기반으로 유저들을 연령대별로 그룹화 한 후 tf-idf가 높은 단어(키워드)를 각 그룹을 대표하는 키워드로 선정한다. 해당 키워드를 통해 그룹 내 유저들에게 '추천도서', '추천문구', '페이지'를 추천해주는 서비스를 제안한다. 

<COLAB> 


## Results 

<표> 

20대의 경우, 추천 도서가 모두 자기계발 도서임을 통해 다른 그룹에 비해 자기계발에 관심을 두는 경향이 있다고 볼 수 있으며, 대표 키워드를 '미라클' 또는 '자기계발'로 뽑을 수 있다. 30대의 경우, 추천 도서 모두가 박노해 시인의 시집인 것을 통해 문학에 관심을 두는 경향이 있고 대표 키워드를 '음유시인' 또는 '박노해'라고 할 수 있다. 40대는 추천 도서가 모두 신지식론이며, 대표 키워드를 '크리스찬' 혹은 '독실한 신자'가 될 것이다. 마지막으로 50대 이상의 연령층에서는 추천 스크랩 문구에 공통적으로 '나이들수록'이라는 단어가 보여지고 따라서 '위로', '성찰'과 같은 단어들을 키워드로 제안할 수 있을 것이다. 


## Limitation 

- 도서 내 추출된 스크랩 문구와 유저가 자체적으로 생성한 메모 형식의 스크랩 데이터 분리의 필요성 
- 자연어 처리 과정 중 한글 텍스트 정제의 한계 

---

# 2. 책 카테고리 별 북클럽 책 추천 서비스 

## Purpose 

북클럽 서비스는 독서 경험이 부족한 독자들에게 적합한 책을 고를 수 있도록 카테고리 별로 선정된 다양한 책들을 소개하며, 읽고 싶은 책을 찾게 된 독자들이 공통의 책을 찾은 모임에 가입 할 수 있게 한다. 하지만 2023년 6월 기준 북클럽 서비스는 2022년 3월 이후로 잠정 연기되어 있다. 이를 위해 북클럽 서비스를 발전시키고자, 머신러닝을 포함한 추천 알고리즘을 도입하고자 한다. 



## Processing 




## Results 

| book_id | title                                                             |
| ------- | ----------------------------------------------------------------- |
| 626     | 돈, 뜨겁게 사랑하고 차갑게 다루어라                                              |
| 3382    | 왜 주식인가 - 부자가 되려면 자본이 일하게 하라                                       |
| 4337    | 워런 버핏과의 점심식사 - 가치투자자로 거듭나다                                        |
| 402481  | 현명한 투자자 - 벤저민 그레이엄 직접 쓴 마지막 개정판, 개정4판                             |
| 135689  | 네이버 증권으로 배우는 주식투자 실전 가이드북 - 주식 고수들만 아는 '네이버 증권 200% 활용법!', 개정증보판  |
| 625     | 전설로 떠나는 월가의 영웅 - 13년간 주식으로 단 한 해도 손실을 본 적이 없는 피터린치 투자, 2017 최신개정판 |
| 17143   | 투자에 대한 생각 - 월스트리트가 가장 신뢰한 하워드 막스의 20가지 투자 철학                      |
| 506     | 위대한 기업에 투자하라                                                      |
| 1721576 | 강방천 & 존리와 함께하는 나의 첫 주식 교과서 - 기본부터 제대로 배우는 평생 투자 원칙                |
| 121002  | 기업공시 완전정복 - 경영 전략과 투자의 항방이 한눈에 보이는                                |

로지스텍 회귀 모델 성능에 영향을 미치는 변수는 완독률과 평균평점이며, 리더스 책 카테고리 별 분류 데이터를 'KoNLP' 패키지를 통해 유저의 관심분야에 맞게 재분류하여 책을 독자에게 추천할 수 있게 한다. 위 표는 재테크와 관련된 상위 10개 항목을 추천한 리스트이다. 



## Limitation 

- 머신러닝 알고리즘의 낮은 specificity 
- 자연어 처리 과정 중 한글 텍스트 정제의 한계 
- 특정 분야의 책들에 편중된 예측값 
- 기존 리더스 어플 내 이용자 관심 분야 세분화 부족 
