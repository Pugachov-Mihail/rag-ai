
package main
import "fmt"

type Database struct { conn string }

func (db *Database) Connect() {
    fmt.Println("Connecting...")
}

func (db *Database) Query(sql string) {
    db.Connect()
    fmt.Println("Executing:", sql)
}

func ExecuteWorkflow(db *Database) {
    db.Query("SELECT 1")
    fmt.Println("Workflow done")
}
