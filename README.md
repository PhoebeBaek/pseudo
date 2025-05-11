## DiningAI

### 작동 flow
0. `streamlit run streamlit_demo.py` 코드로 streamlit UI 실행
2. UI 내에 이미지 업로드
3. menu_analysis_agent가 input_image MCP tool을 활용해 업로드된 이미지 분석 및 필요한 식료품 반환
4. items_query_agent가 mongodb_query MCP tool을 활용해 해당 식료품을 MongoDB에서 서치
