import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components" as Components

ApplicationWindow {
    id: root
    width: 1180
    height: 760
    visible: true
    title: "Reference Review Component Gallery"
    color: "#f7f7f2"

    GridLayout {
        anchors.fill: parent
        anchors.margins: 16
        columns: 2
        columnSpacing: 12
        rowSpacing: 12

        Components.MarkdownPanel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Overflow Context"
            markdownText: "# Synthetic Fixture\n\n" + "Long context ".repeat(80)
            blockedMessage: "1 external link blocked"
        }

        Components.ArtifactPanel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            artifactCount: 4
        }

        Components.NotesPanel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            noteText: "Synthetic review note"
            saveFailure: "Synthetic save failure"
        }

        Components.CodexPanel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            streamText: "Synthetic streamed response"
            refusalText: "Policy block"
            candidateCount: 2
        }
    }
}
