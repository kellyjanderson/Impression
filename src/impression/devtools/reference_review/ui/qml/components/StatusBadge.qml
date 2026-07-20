import QtQuick

Rectangle {
    id: root
    property string label: ""
    property string tone: "neutral"

    implicitHeight: 28
    implicitWidth: Math.max(96, labelText.implicitWidth + 20)
    radius: 4
    color: root.tone === "warning" ? "#ffe2b8" : root.tone === "ready" ? "#d6eeee" : "#ecece4"
    border.color: root.tone === "warning" ? "#d9a457" : root.tone === "ready" ? "#90bdc1" : "#c9c8be"

    Text {
        id: labelText
        anchors.fill: parent
        anchors.leftMargin: 10
        anchors.rightMargin: 10
        text: root.label
        color: root.tone === "warning" ? "#5c360f" : root.tone === "ready" ? "#1f4f52" : "#42453f"
        font.pixelSize: 12
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
