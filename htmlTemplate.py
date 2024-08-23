css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.dribbble.com/userupload/3668818/file/original-82d33fec976d80ee37bdc10eb32c87c2.png?resize=752x">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

prompt_template = Template("""
        Provide an ANSWER to the ACTUAL QUESTION
        Make sure to provide your ANSWER following the EXAMPLE ANSWER FORMAT below:

        EXAMPLE ANSWER FORMAT 1:
        QUESTION: What is LTA?
        ANSWER: Thank you for the relevant question. LTA is a government organisation based on.... Source: LTA Annual Report, Pg. 1


        Now answer this ACTUAL QUESTION in the same way:
        QUESTION: $question
        """)prompt_template = Template("""
        Provide an ANSWER to the ACTUAL QUESTION
        Make sure to provide your ANSWER following the EXAMPLE ANSWER FORMAT below:

        EXAMPLE ANSWER FORMAT 1:
        QUESTION: What is LTA?
        ANSWER: Thank you for the relevant question. LTA is a government organisation based on.... Source: LTA Annual Report, Pg. 1


        Now answer this ACTUAL QUESTION in the same way:
        QUESTION: $question
        """)