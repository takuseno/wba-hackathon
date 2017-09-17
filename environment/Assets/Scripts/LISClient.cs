using System;
using System.Collections.Generic;
using CI.HttpClient;
using UnityEngine;
using System.Text;


public class LISClient {
    private HttpClient client = new HttpClient();
    private Queue<string> queue = new Queue<string>(){};

    public string host = "localhost";
    public string port = "8765";

    public bool HasAction = false;
    public bool Calling = false;
    public int latestScene = -1;

    private Uri createUri;
    private Uri stepUri;
    private Uri resetUri;
    private Uri flushUri;
    private Uri stageUri;

    public LISClient(string identifier) {
        UnityEngine.Debug.Log("Identifier: " + identifier);
        createUri = new Uri("http://" + host + ":" + port + "/create/" + identifier);
        stepUri   = new Uri("http://" + host + ":" + port + "/step/" + identifier);
        resetUri  = new Uri("http://" + host + ":" + port + "/reset/" + identifier);
        flushUri  = new Uri("http://" + host + ":" + port + "/flush/" + identifier);
        stageUri  = new Uri("http://" + host + ":" + port + "/update_stage/" + identifier);
    }

    public string GetAction() {
        string action = queue.Dequeue();
        HasAction = (queue.Count > 0);
        return action;
    }

    void Call(Uri uri, byte[] payload) {
        Calling = true;
        client.Post(uri, new ByteArrayContent(payload, "text/plain"), (r) => {
                string rawData = r.Data;
                string[] data = rawData.Split(new Char[] {'/'});
                // Action
                queue.Enqueue(data[0]);
                // Scene number
                if (data.Length > 1) {
                    Int32.TryParse(data[1], out latestScene);
                }
                HasAction = true;
                Calling = false;
            });
    }

    public void Create(byte[] payload) {
        Call(createUri, payload);
    }

    public void Step(byte[] payload) {
        Call(stepUri, payload);
    }

    public void Reset(byte[] payload) {
        Call(resetUri, payload);
    }

    public void Flush(byte[] payload) {
        Call(flushUri, payload);
    }
}
