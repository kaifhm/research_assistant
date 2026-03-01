const messagesEl = document.getElementById("messages")
const formEl = document.getElementById("form")
const submitBtn = document.getElementById("submit-button")
const messages = []

submitBtn.addEventListener("click", (x) => {
    x.preventDefault()
    let formData = new FormData(formEl)
    let user_message = formData.get('user_message')
    if (user_message === "") {
        return
    }
    formEl.querySelector('textarea').value = ""
    messages.push({"role": "human", "content": user_message})
    formData.append("messages", JSON.stringify(messages))
    
    submitBtn.disabled = true
    let newMessageNode = document.createElement('p')
    newMessageNode.classList.add("message")

    let humanMessage = newMessageNode.cloneNode()
    humanMessage.textContent = user_message
    messagesEl.appendChild(humanMessage)

    newMessageNode.classList.add("ai-message")
    messagesEl.appendChild(newMessageNode)
    
    async function fetchAndStream(url, formData) {
        const response = await fetch(url, {
            method: "POST",
            body: formData,
        });
        

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let message = ""

        try {
            while (true) {
            const { done, value } = await reader.read();

            if (done) {
                console.log("Stream complete");
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            newMessageNode.textContent += chunk
            message += chunk
            }
        } finally {
            reader.releaseLock();
            submitBtn.disabled = false
            messages.push({"role": "assistant", "content": message})
        }

    }

    fetchAndStream(formEl.action, formData)
})