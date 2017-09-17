using UnityEngine;
using UnityEditor.SceneManagement;
using MsgPack;

[RequireComponent(typeof (AgentController))]
[RequireComponent(typeof (AgentSensor))]
public class AgentBehaviour : MonoBehaviour {
    static public string identifier = "myagent_" + System.Guid.NewGuid();
    static private LISClient client = new LISClient(identifier);

    private AgentController controller;
    private AgentSensor sensor;

    private MsgPack.CompiledPacker packer = new MsgPack.CompiledPacker();

    bool created = false;

    string lastAction = "-1";

    void OnCollisionEnter(Collision col) {
        if(col.gameObject.tag == "Reward") {
            NotificationCenter.DefaultCenter.PostNotification(this, "OnRewardCollision");
        }
    }

    int GetSceneNum() {
        return PlayerPrefs.GetInt("Scene Number") + 1;
    }

    byte[] GenerateMessage() {
        Message msg = new Message();

        msg.reward = PlayerPrefs.GetFloat("Reward");
        msg.image = sensor.GetRgbImages();
        msg.depth = sensor.GetDepthImages();
        msg.scene_num = GetSceneNum();  // Scene number

        switch(lastAction) {
        case "0":
            msg.rotation = PlayerPrefs.GetFloat("Rotation Speed");
            break;
        case "1":
            msg.rotation = -PlayerPrefs.GetFloat("Rotation Speed");
            break;
        case "2":
            msg.movement = PlayerPrefs.GetFloat("Movement Speed");
            break;
        default:
            break;
        }

        return packer.Pack(msg);
    }

    byte[] GenerateResetMessage(bool finished) {
        ResetMessage msg = new ResetMessage();

        msg.reward = PlayerPrefs.GetFloat("Reward");
        msg.success = PlayerPrefs.GetInt("Success Count");
        msg.failure = PlayerPrefs.GetInt("Failure Count");
        msg.elapsed = PlayerPrefs.GetInt("Elapsed Time");
        msg.finished = finished;

        return packer.Pack(msg);
    }

    public void Reset() {
        client.Reset(GenerateResetMessage(false));
    }

    public void Finish() {
        client.Reset(GenerateResetMessage(true));
    }

    void Start () {
        controller = GetComponent<AgentController>();
        sensor = GetComponent<AgentSensor>();
    }

    void LateUpdate () {
        // Go to next stage if needed
        int sceneNum = GetSceneNum();
        UnityEngine.Debug.Log("local " + sceneNum);
        UnityEngine.Debug.Log("global " + client.latestScene);
        if (sceneNum < client.latestScene) {
            UnityEngine.Debug.Log("Stage skip -> " + client.latestScene);
            Finish();
            PlayerPrefs.SetInt("Success Count", 0);
            PlayerPrefs.SetInt("Failure Count", 0);
            EditorSceneManager.LoadScene(Scenes.Next());
        }


        if(!created) {
            if(!client.Calling) {
                client.Create(GenerateMessage());
                created = true;
            }
        } else {
            if(client.HasAction) {
                lastAction = client.GetAction();
                controller.PerformAction(lastAction);
            }
            if(!client.Calling) {
                client.Step(GenerateMessage());
            }
        }
    }
}
