"""
Gradio Demo for Dual-Task ABSA Model
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import gradio as gr

# Handle imports from different directories
try:
    from model_service import get_model_service
except ImportError:
    # Try relative import if running as module
    from .model_service import get_model_service

# Initialize model service
print("Initializing model service for Gradio demo...")
model_service = get_model_service()


def predict_sentiment(text):
    """
    Predict sentiment cho text và format kết quả cho Gradio
    
    Args:
        text: Input text
    
    Returns:
        str: Formatted prediction results
    """
    if not text or not text.strip():
        return "Vui lòng nhập text để phân tích!"
    
    try:
        result = model_service.predict(text, filter_absent=True, top_k=5)
        
        # Format output
        output = f"**Text:** {result['text']}\n\n"
        output += "**Kết quả phân tích (Dual-Task):**\n\n"
        
        if not result['predictions']:
            output += "Không phát hiện aspect nào trong text này."
            return output
        
        for aspect, pred in result['predictions'].items():
            present = "✅ CÓ" if pred['present'] else "❌ KHÔNG"
            present_conf = pred['present_confidence'] * 100
            sentiment = pred['sentiment'].upper()
            sentiment_conf = pred['sentiment_confidence'] * 100
            
            # Emoji based on sentiment
            if sentiment == 'POSITIVE':
                emoji = "✅"
            elif sentiment == 'NEGATIVE':
                emoji = "❌"
            else:
                emoji = "➖"
            
            output += f"{emoji} **{aspect}**:\n"
            output += f"   - Phát hiện: {present} (Độ tin cậy: {present_conf:.1f}%)\n"
            output += f"   - Sentiment: {sentiment} (Độ tin cậy: {sentiment_conf:.1f}%)\n"
            output += f"   - Positive: {pred['probabilities']['positive']*100:.1f}%\n"
            output += f"   - Negative: {pred['probabilities']['negative']*100:.1f}%\n"
            output += f"   - Neutral: {pred['probabilities']['neutral']*100:.1f}%\n\n"
        
        return output
    
    except Exception as e:
        return f"Lỗi khi dự đoán: {str(e)}"


# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    # Examples
    examples = [
        "Pin trâu camera xấu",
        "Màn hình đẹp giá rẻ",
        "Hiệu năng tốt nhưng pin nhanh hết",
        "Giao hàng nhanh, đóng gói cẩn thận",
        "Sản phẩm tốt nhưng shop phục vụ chưa tốt"
    ]
    
    # Create interface
    interface = gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(
            label="Nhập text để phân tích sentiment",
            placeholder="Ví dụ: Pin trâu camera xấu",
            lines=3
        ),
        outputs=gr.Markdown(
            label="Kết quả phân tích (Dual-Task)"
        ),
        title="Dual-Task Aspect-Based Sentiment Analysis",
        description="""
        **Dual-Task Learning Approach:**
        - **Task 1: Aspect Detection** - Phát hiện aspects có trong text (binary)
        - **Task 2: Sentiment Classification** - Phân loại sentiment cho các aspects được phát hiện (3-class)
        
        Phân tích sentiment cho 11 aspects: Battery, Camera, Performance, Display, Design, 
        Packaging, Price, Shop_Service, Shipping, General, Others
        
        Mỗi aspect sẽ được dự đoán: Có/Không có trong text, và nếu có thì sentiment là gì (Positive, Negative, Neutral)
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface


if __name__ == "__main__":
    # Launch Gradio interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

