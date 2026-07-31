from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return "EstateIQ Flask Server Running"


@app.route("/vapi/webhook", methods=["POST"])
def vapi_webhook():

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "no data received"
        })


    message = data.get("message", {})

    event_type = message.get("type")

    print("\n========== VAPI EVENT ==========")
    print("EVENT TYPE:", event_type)


    # Only capture completed calls
    if event_type == "end-of-call-report":

        print("\n========== END CALL REPORT ==========")

        # Print only the message section
        print(message)

        print("====================================")


    else:
        print("Ignoring event:", event_type)


    return jsonify({
        "status": "received"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )